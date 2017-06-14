package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_test(
    name = "iterator_ops_test",
    size = "small",
    srcs = ["iterator_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "batch_dataset_op_test",
    size = "small",
    srcs = ["batch_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:string_ops",
    ],
)

py_test(
    name = "bucketing_test",
    size = "small",
    srcs = ["bucketing_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:string_ops",
    ],
)

py_test(
    name = "dataset_constructor_op_test",
    size = "small",
    srcs = ["dataset_constructor_op_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "nomac",  # b/62040583
    ],
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "filter_dataset_op_test",
    size = "small",
    srcs = ["filter_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "flat_map_dataset_op_test",
    size = "small",
    srcs = ["flat_map_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "map_dataset_op_test",
    size = "small",
    srcs = ["map_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:string_ops",
    ],
)

py_test(
    name = "range_dataset_op_test",
    size = "small",
    srcs = ["range_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "reader_dataset_ops_test",
    size = "small",
    srcs = ["reader_dataset_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "resample_test",
    size = "medium",
    srcs = ["resample_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "sequence_dataset_op_test",
    size = "small",
    srcs = ["sequence_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "shuffle_dataset_op_test",
    size = "small",
    srcs = ["shuffle_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "cache_dataset_op_test",
    size = "small",
    srcs = ["cache_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "zip_dataset_op_test",
    size = "small",
    srcs = ["zip_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
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
