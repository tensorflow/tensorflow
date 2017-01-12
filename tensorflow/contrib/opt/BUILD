# Description:
#   Optimization routines.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "opt_py",
    srcs = [
        "__init__.py",
        "python/training/external_optimizer.py",
        "python/training/moving_average_optimizer.py",
        "python/training/variable_clipping_optimizer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "@six_archive//:six",
    ],
)

py_test(
    name = "external_optimizer_test",
    srcs = ["python/training/external_optimizer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":opt_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "moving_average_optimizer_test",
    srcs = ["python/training/moving_average_optimizer_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "notsan",  # b/31055119
    ],
    deps = [
        ":opt_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "variable_clipping_optimizer_test",
    srcs = ["python/training/variable_clipping_optimizer_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",  # Flaky: b/29892493
    ],
    deps = [
        ":opt_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
