package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
)

py_library(
    name = "data",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        # The test dependency enables testing batch_dataset_op_test against a pip installation
        "//third_party/tensorflow/contrib/data/python/kernel_tests:dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:dataset_ops",
        "//tensorflow/contrib/data/python/ops:iterator_ops",
        "//tensorflow/contrib/data/python/ops:prefetching_py",
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/contrib/data/python/ops:transformation_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:iterator_ops",
    ],
)

tf_custom_op_library(
    name = "_prefetching_ops.so",
    srcs = [
        "ops/prefetching_ops.cc",
    ],
    deps = [
        "//tensorflow/contrib/data/kernels:prefetching_kernels",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["prefetching_ops"],
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
