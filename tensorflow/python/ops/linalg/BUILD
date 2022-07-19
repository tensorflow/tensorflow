package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "linalg",
    srcs = glob(["*.py"]),
    srcs_version = "PY3",
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
        "//tensorflow/python/ops/signal",
        "//tensorflow/python/trackable:data_structures",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "linalg_impl",
    srcs = ["linalg_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":linear_operator_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:special_math_ops",
    ],
)

py_library(
    name = "linear_operator_util",
    srcs = ["linear_operator_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/module",
    ],
)
