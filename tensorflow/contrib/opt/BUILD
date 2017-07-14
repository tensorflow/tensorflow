# Description:
#   Optimization routines.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "opt_py",
    srcs = [
        "__init__.py",
        "python/training/delay_compensated_gradient_descent.py",
        "python/training/drop_stale_gradient_optimizer.py",
        "python/training/external_optimizer.py",
        "python/training/lazy_adam_optimizer.py",
        "python/training/moving_average_optimizer.py",
        "python/training/nadam_optimizer.py",
        "python/training/variable_clipping_optimizer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "delay_compensated_gradient_descent_test",
    srcs = ["python/training/delay_compensated_gradient_descent_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":opt_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "external_optimizer_test",
    size = "small",
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
    size = "small",
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
        "@six_archive//:six",
    ],
)

py_test(
    name = "variable_clipping_optimizer_test",
    srcs = ["python/training/variable_clipping_optimizer_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",  # Flaky: b/29892493
        "notap",  # data race due to b/62910646
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

py_test(
    name = "lazy_adam_optimizer_test",
    size = "small",
    srcs = ["python/training/lazy_adam_optimizer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":opt_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "nadam_optimizer_test",
    size = "small",
    srcs = ["python/training/nadam_optimizer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":opt_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "drop_stale_gradient_optimizer_test",
    size = "small",
    srcs = ["python/training/drop_stale_gradient_optimizer_test.py"],
    additional_deps = [
        ":opt_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
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
