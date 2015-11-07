# Description:
# Example neural translation models.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "data_utils",
    srcs = [
        "data_utils.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "seq2seq_model",
    srcs = [
        "seq2seq_model.py",
    ],
    deps = [
        ":data_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/models/rnn:seq2seq",
    ],
)

py_binary(
    name = "translate",
    srcs = [
        "translate.py",
    ],
    deps = [
        ":data_utils",
        ":seq2seq_model",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "translate_test",
    size = "medium",
    srcs = [
        "translate.py",
    ],
    args = [
        "--self_test=True",
    ],
    main = "translate.py",
    deps = [
        ":data_utils",
        ":seq2seq_model",
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
