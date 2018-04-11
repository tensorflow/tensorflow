package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "if_not_windows",
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
        "//tensorflow/contrib/data/python/ops:prefetching_ops",
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/contrib/data/python/ops:shuffle_ops",
        "//tensorflow/contrib/data/python/ops:transformation_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:iterator_ops",
    ],
)

cc_library(
    name = "lib_proto_parsing_for_dataset_ops",
    deps = if_not_windows(["//tensorflow/core:lib_proto_parsing"]),
)

tf_custom_op_library(
    name = "_dataset_ops.so",
    srcs = ["ops/dataset_ops.cc"],
    deps = ["//tensorflow/contrib/data/kernels:dataset_kernels"] +
           if_static(
               extra_deps = [":lib_proto_parsing_for_dataset_ops"],
               otherwise = [],
           ),
)

tf_gen_op_libs(
    op_lib_names = ["dataset_ops"],
)
