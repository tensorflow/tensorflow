package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_custom_op_py_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
    "tf_py_test",
)

py_library(
    name = "kinesis",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_ops",
    ],
)

tf_custom_op_library(
    name = "_dataset_ops.so",
    srcs = ["ops/dataset_ops.cc"],
    deps = [":dataset_kernels"],
)

tf_gen_op_libs(
    op_lib_names = ["dataset_ops"],
)

cc_library(
    name = "dataset_kernels",
    srcs = [
        "kernels/kinesis_dataset_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/platform/s3:aws_crypto",
        "//third_party/eigen3",
        "@aws",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = 1,
)

py_library(
    name = "dataset_ops",
    srcs = [
        "python/ops/kinesis_dataset_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":kinesis_op_loader",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_dataset_ops",
    out = "python/ops/gen_dataset_ops.py",
    deps = ["//tensorflow/contrib/kinesis:dataset_ops_op_lib"],
)

tf_kernel_library(
    name = "dataset_ops_kernels",
    deps = [
        ":dataset_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_custom_op_py_library(
    name = "kinesis_op_loader",
    srcs = ["python/ops/kinesis_op_loader.py"],
    dso = ["//tensorflow/contrib/kinesis:_dataset_ops.so"],
    kernels = [
        ":dataset_ops_kernels",
        "//tensorflow/contrib/kinesis:dataset_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_dataset_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "kinesis_test",
    srcs = ["python/kernel_tests/kinesis_test.py"],
    additional_deps = [
        ":kinesis",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "manual",
        "no_windows",
        "notap",
    ],
)
