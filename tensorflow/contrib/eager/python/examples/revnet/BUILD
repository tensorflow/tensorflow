licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "gpu_py_test")

# Model
py_library(
    name = "ops",
    srcs = ["ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "blocks",
    srcs = ["blocks.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "revnet",
    srcs = ["revnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":blocks",
        "//tensorflow:tensorflow_py",
    ],
)

# Tests
gpu_py_test(
    name = "ops_test",
    size = "large",
    srcs = ["ops_test.py"],
    additional_deps = [
        ":ops",
        "//tensorflow:tensorflow_py",
    ],
)

gpu_py_test(
    name = "blocks_test",
    size = "large",
    srcs = ["blocks_test.py"],
    additional_deps = [
        ":blocks",
        "//tensorflow:tensorflow_py",
    ],
    tags = [
        "optonly",
    ],
)

gpu_py_test(
    name = "revnet_test",
    size = "large",
    srcs = ["revnet_test.py"],
    additional_deps = [
        ":blocks_test",
        ":config",
        ":revnet",
        "//tensorflow:tensorflow_py",
    ],
    tags = [
        "no_pip",  # depends on blocks_test, which is not available in pip package
        "optonly",
    ],
)

# Training
py_library(
    name = "cifar_input",
    srcs = ["cifar_input.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "cifar_tfrecords",
    srcs = ["cifar_tfrecords.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cifar_input",
        ":config",
        ":revnet",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "main_estimator",
    srcs = ["main_estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cifar_input",
        ":main",
        ":revnet",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "main_estimator_lib",
    srcs = ["main_estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cifar_input",
        ":main",
        ":revnet",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "main_estimator_tpu_lib",
    srcs = ["main_estimator_tpu.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cifar_input",
        ":main",
        ":revnet",
        "//tensorflow:tensorflow_py",
    ],
)
