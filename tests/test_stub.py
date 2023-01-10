from ray_detectron2.ray_detectron2 import fib


def test_fib() -> None:
    assert fib(10) == 55
