licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "gpu_py_test")

py_binary(
    name = "densenet",
    srcs = ["densenet.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
    ],
)

gpu_py_test(
    name = "densenet_test",
    size = "large",
    srcs = ["densenet_test.py"],
    additional_deps = [
        ":densenet",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow:tensorflow_py",
    ],
    tags = [
        "no_pip",
        "optonly",
    ],
)

cuda_py_test(
    name = "densenet_graph_test",
    size = "large",
    srcs = ["densenet_graph_test.py"],
    additional_deps = [
        ":densenet",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
    tags = [
        "no_pip",
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
    ],
)
