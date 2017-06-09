# Description:
#   Contains packages used for creating and loading datasets.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "data",
    deps = [
        ":dataset",
        ":dataset_data_provider",
        ":parallel_reader",
        ":prefetch_queue",
        ":tfexample_decoder",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "data_pip",
    deps = [
        ":data",
        ":test_utils",
    ],
)

py_library(
    name = "data_decoder",
    srcs = ["data_decoder.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "data_provider",
    srcs = ["data_provider.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "dataset_data_provider",
    srcs = ["dataset_data_provider.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":data_provider",
        ":parallel_reader",
    ],
)

py_test(
    name = "dataset_data_provider_test",
    srcs = ["dataset_data_provider_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset",
        ":dataset_data_provider",
        ":test_utils",
        ":tfexample_decoder",
        "//tensorflow/contrib/slim:queues",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "parallel_reader",
    srcs = ["parallel_reader.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "parallel_reader_test",
    size = "small",
    srcs = ["parallel_reader_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":parallel_reader",
        ":test_utils",
        "//tensorflow/contrib/slim:queues",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "prefetch_queue",
    srcs = ["prefetch_queue.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "prefetch_queue_test",
    size = "small",
    srcs = ["prefetch_queue_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":prefetch_queue",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:lib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tfexample_decoder",
    srcs = ["tfexample_decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":data_decoder",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_ops",
    ],
)

py_test(
    name = "tfexample_decoder_test",
    srcs = ["tfexample_decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tfexample_decoder",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
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
