# Description:
#   Contains the legacy TF layers (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "layers_base",
    srcs = [
        "__init__.py",
        "base.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/mixed_precision/experimental:policy",
        "//tensorflow/python/training/tracking:base",
    ],
)

py_library(
    name = "convolutional",
    srcs = ["convolutional.py"],
    deps = [
        ":layers_base",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/keras/layers",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    deps = [
        ":layers_base",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/keras/layers",
    ],
)

py_library(
    name = "normalization",
    srcs = ["normalization.py"],
    deps = [
        ":layers_base",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/keras/layers:normalization",
    ],
)

py_library(
    name = "pooling",
    srcs = ["pooling.py"],
    deps = [
        ":layers_base",
        "//tensorflow/python:util",
        "//tensorflow/python/keras/layers",
    ],
)

tf_py_test(
    name = "base_test",
    size = "small",
    srcs = ["base_test.py"],
    main = "base_test.py",
    python_version = "PY3",
    deps = [
        ":core",
        ":layers_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/engine:input_spec",
    ],
)

tf_py_test(
    name = "core_test",
    size = "small",
    srcs = ["core_test.py"],
    main = "core_test.py",
    python_version = "PY3",
    deps = [
        ":core",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:combinations",
    ],
)

tf_py_test(
    name = "convolutional_test",
    size = "small",
    srcs = ["convolutional_test.py"],
    main = "convolutional_test.py",
    python_version = "PY3",
    deps = [
        ":convolutional",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "pooling_test",
    size = "small",
    srcs = ["pooling_test.py"],
    main = "pooling_test.py",
    python_version = "PY3",
    tags = ["no_rocm"],
    deps = [
        ":pooling",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:random_ops",
    ],
)

cuda_py_test(
    name = "normalization_test",
    size = "medium",
    srcs = ["normalization_test.py"],
    main = "normalization_test.py",
    python_version = "PY3",
    shard_count = 10,
    deps = [
        ":convolutional",
        ":normalization",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)
