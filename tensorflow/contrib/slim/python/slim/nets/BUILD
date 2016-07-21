# Description:
#   Contains typical networks definitions.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "nets",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":alexnet",
        ":overfeat",
        ":vgg",
    ],
)

py_library(
    name = "alexnet",
    srcs = ["alexnet.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "alexnet_test",
    size = "medium",
    srcs = ["alexnet_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":alexnet",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "overfeat",
    srcs = ["overfeat.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "overfeat_test",
    size = "medium",
    srcs = ["overfeat_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":overfeat",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "vgg",
    srcs = ["vgg.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "vgg_test",
    size = "medium",
    srcs = ["vgg_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":vgg",
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
