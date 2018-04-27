# Description:
# Example TensorFlow models for MNIST used in tutorials

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":input_data",
        ":mnist",
    ],
)

py_library(
    name = "input_data",
    srcs = ["input_data.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "mnist",
    srcs = [
        "mnist.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "fully_connected_feed",
    srcs = [
        "fully_connected_feed.py",
    ],
    srcs_version = "PY2AND3",
    tags = ["optonly"],
    deps = [
        ":input_data",
        ":mnist",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "mnist_with_summaries",
    srcs = [
        "mnist_with_summaries.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":input_data",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "mnist_softmax",
    srcs = [
        "mnist_softmax.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":input_data",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "mnist_deep",
    srcs = [
        "mnist_deep.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":input_data",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "fully_connected_feed_test",
    size = "small",
    srcs = [
        "fully_connected_feed.py",
    ],
    args = [
        "--fake_data",
        "--max_steps=10",
    ],
    main = "fully_connected_feed.py",
    srcs_version = "PY2AND3",
    deps = [
        ":input_data",
        ":mnist",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "mnist_with_summaries_test",
    size = "small",
    srcs = [
        "mnist_with_summaries.py",
    ],
    args = [
        "--fake_data",
        "--max_steps=10",
        "--learning_rate=0.00",
    ],
    main = "mnist_with_summaries.py",
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # http://b/29184009
    deps = [
        ":input_data",
        "//tensorflow:tensorflow_py",
    ],
)
