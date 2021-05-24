package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "signal",
    srcs = [
        "dct_ops.py",
        "fft_ops.py",
        "mel_ops.py",
        "mfcc_ops.py",
        "reconstruction_ops.py",
        "shape_ops.py",
        "signal.py",
        "spectral_ops.py",
        "util_ops.py",
        "window_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:spectral_ops_gen",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)
