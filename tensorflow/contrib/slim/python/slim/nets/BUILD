# Description:
#   Contains typical networks definitions.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow_models:__subpackages__",
    ],
)

py_library(
    name = "nets",
    deps = [
        ":alexnet",
        ":inception",
        ":overfeat",
        ":resnet_v1",
        ":resnet_v2",
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
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "inception",
    srcs = ["inception.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_v1",
        ":inception_v2",
        ":inception_v3",
    ],
)

py_library(
    name = "inception_v1",
    srcs = ["inception_v1.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "inception_v2",
    srcs = ["inception_v2.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "inception_v3",
    srcs = ["inception_v3.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "inception_v1_test",
    size = "large",
    srcs = ["inception_v1_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "inception_v2_test",
    size = "large",
    srcs = ["inception_v2_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "inception_v3_test",
    size = "large",
    srcs = ["inception_v3_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
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
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "resnet_utils",
    srcs = ["resnet_utils.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "resnet_v1",
    srcs = ["resnet_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
    ],
)

py_test(
    name = "resnet_v1_test",
    size = "medium",
    srcs = ["resnet_v1_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "resnet_v2",
    srcs = ["resnet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
    ],
)

py_test(
    name = "resnet_v2_test",
    size = "medium",
    srcs = ["resnet_v2_test.py"],
    srcs_version = "PY2AND3",
    deps = [
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
