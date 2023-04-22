load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

licenses(["notice"])

py_test(
    name = "public_symbol_test",
    srcs = ["public_symbol_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "np_config_test",
    srcs = ["np_config_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//third_party/py/numpy",
    ],
)
