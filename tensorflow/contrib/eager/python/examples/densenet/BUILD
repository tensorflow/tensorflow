licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_binary")

py_binary(
    name = "densenet",
    srcs = ["densenet.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
    ],
)

cuda_py_test(
    name = "densenet_test",
    size = "medium",
    srcs = ["densenet_test.py"],
    additional_deps = [
        ":densenet",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow:tensorflow_py",
    ],
    shard_count = 4,
    tags = [
        "no_pip",
        "optonly",
        "oss_serial",
    ],
)

cuda_py_test(
    name = "densenet_graph_test",
    size = "medium",
    srcs = ["densenet_graph_test.py"],
    additional_deps = [
        ":densenet",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
    shard_count = 4,
    tags = [
        "no_pip",
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
        "oss_serial",
    ],
)
