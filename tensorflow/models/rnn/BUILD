# Description:
# Example RNN models, including language models and sequence-to-sequence models.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("/tensorflow/tensorflow", "cuda_py_tests")

py_library(
    name = "linear",
    srcs = [
        "linear.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "linear_test",
    size = "small",
    srcs = ["linear_test.py"],
    deps = [
        ":linear",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "rnn_cell",
    srcs = [
        "rnn_cell.py",
    ],
    deps = [
        ":linear",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "rnn_cell_test",
    size = "small",
    srcs = ["rnn_cell_test.py"],
    deps = [
        ":rnn_cell",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
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
    deps = [
        ":rnn_cell",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_tests(
    name = "rnn_tests",
    srcs = [
        "rnn_test.py",
    ],
    additional_deps = [
        ":rnn",
    ],
)

py_library(
    name = "seq2seq",
    srcs = [
        "seq2seq.py",
    ],
    deps = [
        ":rnn",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "seq2seq_test",
    srcs = [
        "seq2seq_test.py",
    ],
    deps = [
        ":seq2seq",
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
