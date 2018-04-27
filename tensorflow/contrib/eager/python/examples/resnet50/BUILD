licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "resnet50",
    srcs = ["resnet50.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
    ],
)

py_library(
    name = "resnet50_test_lib",
    srcs = ["resnet50_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet50",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
    ],
)

cuda_py_test(
    name = "resnet50_test",
    size = "large",
    srcs = ["resnet50_test.py"],
    additional_deps = [
        ":resnet50",
        "//tensorflow/contrib/summary:summary_test_util",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "resnet50_graph_test",
    size = "large",
    srcs = ["resnet50_graph_test.py"],
    additional_deps = [
        ":resnet50",
        "//tensorflow/contrib/summary:summary_test_util",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
)
