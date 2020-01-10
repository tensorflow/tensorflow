# Description:
# Example TensorFlow models for CIFAR-10

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "cifar10_input",
    srcs = ["cifar10_input.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "cifar10_input_test",
    srcs = ["cifar10_input_test.py"],
    deps = [
        ":cifar10_input",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "cifar10",
    srcs = ["cifar10.py"],
    deps = [
        ":cifar10_input",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "cifar10_eval",
    srcs = [
        "cifar10_eval.py",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":cifar10",
    ],
)

py_binary(
    name = "cifar10_train",
    srcs = [
        "cifar10_train.py",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":cifar10",
    ],
)

py_binary(
    name = "cifar10_multi_gpu_train",
    srcs = [
        "cifar10_multi_gpu_train.py",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":cifar10",
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
    visibility = ["//tensorflow:__subpackages__"],
)
