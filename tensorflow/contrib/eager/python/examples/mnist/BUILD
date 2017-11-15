licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_binary(
    name = "mnist",
    srcs = ["mnist.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow/examples/tutorials/mnist:input_data",
    ],
)

cuda_py_test(
    name = "mnist_test",
    srcs = ["mnist_test.py"],
    additional_deps = [
        ":mnist",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "mnist_graph_test",
    srcs = ["mnist_graph_test.py"],
    additional_deps = [
        ":mnist",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
)
