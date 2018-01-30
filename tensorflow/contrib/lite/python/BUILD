licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "lite",
    srcs = ["lite.py"],
    # data = [
    #     "//tensorflow/contrib/lite/toco/python:toco_from_protos",
    # ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":op_hint",
        "//tensorflow/contrib/lite/toco:model_flags_proto_py",
        "//tensorflow/contrib/lite/toco:toco_flags_proto_py",
        "//tensorflow/contrib/lite/toco/python:tensorflow_wrap_toco",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "op_hint",
    srcs = ["op_hint.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "lite_test",
    srcs = ["lite_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_oss"],
    deps = [
        ":lite",
        ":op_hint",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
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
