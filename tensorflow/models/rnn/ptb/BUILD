# Description:
# Python support for TensorFlow.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "reader",
    srcs = ["reader.py"],
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "reader_test",
    srcs = ["reader_test.py"],
    deps = [
        ":reader",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "ptb_word_lm",
    srcs = [
        "ptb_word_lm.py",
    ],
    deps = [
        ":reader",
        "//tensorflow:tensorflow_py",
        "//tensorflow/models/rnn",
        "//tensorflow/models/rnn:rnn_cell",
        "//tensorflow/models/rnn:seq2seq",
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
