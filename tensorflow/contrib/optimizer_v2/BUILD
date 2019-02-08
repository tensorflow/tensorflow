# Prototype of OptimizerV2.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "optimizer_v2_py",
    srcs = ["optimizer_v2_symbols.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":training",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "training",
    srcs = [
        "adadelta.py",
        "adagrad.py",
        "adam.py",
        "gradient_descent.py",
        "momentum.py",
        "optimizer_v2.py",
        "rmsprop.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
    ],
)

cuda_py_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    additional_deps = [
        ":training",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "adagrad_test",
    size = "small",
    srcs = ["adagrad_test.py"],
    additional_deps = [
        ":training",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "adam_test",
    size = "small",
    srcs = ["adam_test.py"],
    additional_deps = [
        ":training",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "checkpointable_utils_test",
    srcs = ["checkpointable_utils_test.py"],
    additional_deps = [
        ":training",
        "@six_archive//:six",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:layers_base",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/keras",
    ],
    tags = ["notsan"],
)

cuda_py_test(
    name = "gradient_descent_test",
    size = "medium",
    srcs = ["gradient_descent_test.py"],
    additional_deps = [
        ":training",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:resources",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "momentum_test",
    size = "medium",
    srcs = ["momentum_test.py"],
    additional_deps = [
        ":training",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:resources",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "optimizer_v2_test",
    size = "medium",
    srcs = ["optimizer_v2_test.py"],
    additional_deps = [
        ":training",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "rmsprop_test",
    size = "small",
    srcs = ["rmsprop_test.py"],
    additional_deps = [
        ":training",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
    tags = ["optonly"],
)
