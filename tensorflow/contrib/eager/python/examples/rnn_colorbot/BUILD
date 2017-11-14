licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_binary(
    name = "rnn_colorbot",
    srcs = ["rnn_colorbot.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "rnn_colorbot_test",
    srcs = ["rnn_colorbot_test.py"],
    additional_deps = [
        ":rnn_colorbot",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow:tensorflow_py",
    ],
)
