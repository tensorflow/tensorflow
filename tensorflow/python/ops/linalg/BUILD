package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "linalg",
    srcs = glob(["*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":linalg_impl",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "linalg_impl",
    srcs = ["linalg_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:special_math_ops",
    ],
)
