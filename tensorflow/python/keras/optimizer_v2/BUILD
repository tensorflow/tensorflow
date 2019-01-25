# Description:
#   Contains the Keras OptimizerV2 API (internal TensorFlow version).

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "optimizer_v2",
    srcs = [
        "adadelta.py",
        "adagrad.py",
        "adam.py",
        "adamax.py",
        "ftrl.py",
        "gradient_descent.py",
        "nadam.py",
        "optimizer_v2.py",
        "rmsprop.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:distribute",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/keras:backend_config",
    ],
)

cuda_py_test(
    name = "adagrad_test",
    size = "medium",
    srcs = ["adagrad_test.py"],
    additional_deps = [
        ":optimizer_v2",
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
    shard_count = 4,
)

cuda_py_test(
    name = "adam_test",
    size = "medium",
    srcs = ["adam_test.py"],
    additional_deps = [
        ":optimizer_v2",
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
    shard_count = 4,
)

cuda_py_test(
    name = "adamax_test",
    size = "medium",
    srcs = ["adamax_test.py"],
    additional_deps = [
        ":optimizer_v2",
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
    shard_count = 4,
)

cuda_py_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    additional_deps = [
        ":optimizer_v2",
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
    shard_count = 4,
)

cuda_py_test(
    name = "ftrl_test",
    size = "medium",
    srcs = ["ftrl_test.py"],
    additional_deps = [
        ":optimizer_v2",
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
    shard_count = 4,
)

cuda_py_test(
    name = "gradient_descent_test",
    size = "medium",
    srcs = ["gradient_descent_test.py"],
    additional_deps = [
        ":optimizer_v2",
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
    shard_count = 4,
)

cuda_py_test(
    name = "nadam_test",
    size = "medium",
    srcs = ["nadam_test.py"],
    additional_deps = [
        ":optimizer_v2",
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
    shard_count = 4,
)

py_test(
    name = "optimizer_v2_test",
    size = "medium",
    srcs = ["optimizer_v2_test.py"],
    shard_count = 8,
    tags = [
        "no_windows",
    ],
    deps = [
        ":optimizer_v2",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "rmsprop_test",
    size = "medium",
    srcs = ["rmsprop_test.py"],
    additional_deps = [
        ":optimizer_v2",
        "//tensorflow/python/eager:def_function",
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
    shard_count = 2,
)
