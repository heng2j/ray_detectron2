""" Running Detectron2 with Ray Demo """

# Import  some common libraries
import json
import logging
import os
import time
from typing import Dict
import numpy as np
import cv2

# Import Detectron2 related libraries
import detectron2
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine import DefaultTrainer, hooks
from detectron2.structures import BoxMode
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

# Import Ray related libraries
import ray
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner

# Import Torch related libraries
import torch
from torch import nn
from torch.utils.data import DataLoader

setup_logger()

# Santity check on Torch version and CUDA avability
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_dataset_dicts(img_dir: str) -> Dict:
    """Generage dataset and annotation dictionary from given image diretory and labels file

    Args:
        img_dir (str): directory path for images

    Returns:
        Dict: Dataset and annotation dictionary
    """

    dataset_type = img_dir.split("/")[-1]
    json_file = os.path.join(img_dir, f"labels_{dataset_type}.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    starting_index = 1
    if dataset_type == "train":
        starting_index = 62

    annos = {i + starting_index: [] for i in range(len(imgs_anns["images"]))}

    for ann in imgs_anns["annotations"]:
        annos[ann["image_id"]].append(ann)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns["images"]):
        record = {}

        filename = os.path.join(img_dir, v["file_name"])
        (height, width) = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = v["id"]
        record["height"] = height
        record["width"] = width

        objs = []
        for anno in annos[record["image_id"]]:

            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": anno["segmentation"],
                "category_id": anno["category_id"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def validate_epoch(validation_loader: DataLoader, model: nn.modules, trainer: DefaultTrainer) -> None:
    """Run validation with the training model on the entire validation dataset

    Args:
        validation_loader (DataLoader): validation dataloader
        model (nn.modules): training model
        trainer (DefaultTrainer): Detectron2 trainer
    """

    val_losses = []

    for idx, inputs in enumerate(validation_loader):

        # How loss is calculated on train_loop
        metrics_dict = model(inputs)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v) for k, v in metrics_dict.items()
        }
        loss_batch = sum(loss for loss in metrics_dict.values())
        val_losses.append(loss_batch)

    mean_loss = np.mean(val_losses)
    trainer.storage.put_scalar("validation_loss", mean_loss)


def train_func(config: Dict) -> None:
    """Training function for Ray's TorchTrainer

    Args:
        config (Dict): Training configuration dictionary

    """

    # import get_cfg, MetadataCatalog and DatasetCatalog as mentioned in:
    # https://github.com/ray-project/ray/issues/15946
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog

    # Register Train and Val datasets
    img_dir = config.get("img_dir")

    for d in ["train", "val"]:
        DatasetCatalog.register("license_plates_" + d, lambda d=d: get_dataset_dicts(os.path.join(img_dir, d)))
        MetadataCatalog.get("license_plates_" + d).set(thing_classes=["license_plates"])

    # Set defult training params
    epochs = config.get("epochs", 3)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = "./detector_output"
    cfg.DATASETS.TRAIN = ("license_plates_train",)
    cfg.DATASETS.TEST = ("license_plates_val",)
    cfg.TEST.EVAL_PERIOD = 40
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = config.get(
        "batch_size", 2
    )  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (Vehicle registration plate). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # Set up Detectron2 trainer
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.iter = starting_epoch = 0

    # Get model, optimizer, dataloaders from trainer
    model = trainer._trainer.model
    optimizer = trainer._trainer.optimizer
    train_data_loader = trainer._trainer.data_loader
    train_data_loader_iter_obj = iter(train_data_loader)
    val_data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], DatasetMapper(cfg, True))

    # Start the training loop
    with EventStorage(starting_epoch) as trainer.storage:
        try:
            trainer.before_train()
            for epoch in range(starting_epoch, epochs):

                model.train()

                # trainer.run_step()
                # Decomposed the run step from SimpleTrainer
                start = time.perf_counter()
                data = next(train_data_loader_iter_obj)
                data_time = time.perf_counter() - start

                loss_dict = model(data)

                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

                    optimizer.zero_grad()
                    losses.backward()

                    trainer._trainer._write_metrics(loss_dict, data_time)

                    optimizer.step()

                # Set up checkpoint
                checkpoint = None
                if epoch % config.get("checkpoint_frequency") == 0:
                    checkpoint = Checkpoint.from_dict(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),  # FIXME - only work for single worker aka single GPU
                            # "model": model.module.state_dict(),
                            # if isinstance(model, DistributedDataParallel)
                            # else model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }
                    )

                # Eval & Reporting
                if (epoch % cfg.TEST.EVAL_PERIOD == 0) or (epoch == 0):
                    validate_epoch(val_data_loader, model, trainer)

                result = {
                    "loss": trainer.storage._latest_scalars["total_loss"][0],
                    "accuracy": trainer.storage._latest_scalars["mask_rcnn/accuracy"][0],
                    "validation_loss": trainer.storage._latest_scalars["validation_loss"][0],
                }

                session.report(result, checkpoint=checkpoint)

                trainer.iter += 1

        finally:
            trainer.after_train()


# Ray Training with Ray Tune
ray.init(log_to_driver=True, logging_level=logging.WARNING)

num_epochs = 1001
smoke_test = False

config = {
    "lr": 1e-3,
    "batch_size": 16,
    "epochs": num_epochs,
    "checkpoint_frequency": 100,
    "img_dir": "/heng/data/FiftyOne_Vehicle_registration_plate_data/data",
}
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
)


tuner = Tuner(
    trainer,
    param_space={"train_loop_config": config},
    tune_config=TuneConfig(
        num_samples=1,
        metric="accuracy",
        mode="max",  # scheduler=pbt_scheduler
    ),
    run_config=RunConfig(
        name="Detector_Training_Demo",
        local_dir="./output/RayDetectoron2/ray_results/",
        stop={"training_iteration": 3 if smoke_test else num_epochs},
        failure_config=FailureConfig(max_failures=3),
        verbose=0,
    ),
)


results = tuner.fit()

print(results.get_best_result(metric="accuracy", mode="max"))
