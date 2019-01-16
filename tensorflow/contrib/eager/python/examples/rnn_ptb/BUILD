licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_binary")

py_binary(
    name = "rnn_ptb",
    srcs = ["rnn_ptb.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/cudnn_rnn:cudnn_rnn_py",
        "//tensorflow/contrib/eager/python:tfe",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "rnn_ptb_test",
    srcs = ["rnn_ptb_test.py"],
    additional_deps = [
        ":rnn_ptb",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "rnn_ptb_graph_test",
    srcs = ["rnn_ptb_graph_test.py"],
    additional_deps = [
        ":rnn_ptb",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
    tags = [
        "oss_serial",
    ],
)
