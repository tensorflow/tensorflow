# Description:
# Example code showing how to add a custom op in tensorflow

package(
    default_visibility = ["//visibility:private"],
    features = ["-parse_headers"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "fact.so",
    srcs = ["fact.cc"],
)

py_library(
    name = "fact_op",
    srcs = ["fact_op.py"],
    data = [":fact.so"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "fact_op_test",
    size = "small",
    srcs = ["fact_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fact_op",
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
    visibility = ["//third_party/tensorflow:__subpackages__"],
)
