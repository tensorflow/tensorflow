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

filegroup(
    name = "test_data",
    srcs = glob(["python/kernel_tests/testdata/*"]),
)

py_library(
    name = "hadoop",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_ops",
    ],
)

tf_custom_op_library(
    name = "_dataset_ops.so",
    srcs = ["ops/dataset_ops.cc"],
    deps = [
        ":dataset_kernels",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["dataset_ops"],
)

cc_library(
    name = "dataset_kernels",
    srcs = ["kernels/hadoop_dataset_ops.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = 1,
)

py_library(
    name = "dataset_ops",
    srcs = [
        "python/ops/hadoop_dataset_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":hadoop_op_loader",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_dataset_ops",
    out = "python/ops/gen_dataset_ops.py",
    deps = ["//tensorflow/contrib/hadoop:dataset_ops_op_lib"],
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
    name = "hadoop_op_loader",
    srcs = ["python/ops/hadoop_op_loader.py"],
    dso = ["//tensorflow/contrib/hadoop:_dataset_ops.so"],
    kernels = [
        ":dataset_ops_kernels",
        "//tensorflow/contrib/hadoop:dataset_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_dataset_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "hadoop_test",
    srcs = ["python/kernel_tests/hadoop_test.py"],
    additional_deps = [
        ":hadoop",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    data = [
        ":test_data",
    ],
    tags = [
        "notap",
    ],
)
