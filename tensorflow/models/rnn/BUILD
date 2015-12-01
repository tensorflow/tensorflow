# Description:
# Example RNN models, including language models and sequence-to-sequence models.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "linear",
    srcs = [
        "linear.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "rnn_cell",
    srcs = [
        "rnn_cell.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":linear",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":rnn",
        ":rnn_cell",
        ":seq2seq",
    ],
)

py_library(
    name = "rnn",
    srcs = [
        "rnn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":rnn_cell",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "seq2seq",
    srcs = [
        "seq2seq.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":rnn",
        "//tensorflow:tensorflow_py",
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
