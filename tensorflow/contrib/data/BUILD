package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "if_static",
)

py_library(
    name = "data",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:dataset_ops",
        "//tensorflow/contrib/data/python/ops:iterator_ops",
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/contrib/data/python/ops:shuffle_ops",
        "//tensorflow/contrib/data/python/ops:transformation_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:iterator_ops",
    ],
)

tf_custom_op_library(
    name = "_dataset_ops.so",
    srcs = ["ops/dataset_ops.cc"],
    deps = ["//tensorflow/contrib/data/kernels:dataset_kernels"] +
           if_static(
               extra_deps = ["//tensorflow/core:lib_proto_parsing"],
               otherwise = [],
           ),
)

tf_gen_op_libs(
    op_lib_names = ["dataset_ops"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        include = [
            "**/*",
        ],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
