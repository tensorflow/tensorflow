licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

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

py_test(
    name = "keras_test",
    srcs = [
        "keras_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "list_literals_test",
    srcs = [
        "list_literals_test.py",
    ],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)
