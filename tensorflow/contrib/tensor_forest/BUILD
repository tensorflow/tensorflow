# TensorFlow code for training random forests.
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//visibility:public",
])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

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

filegroup(
    name = "custom_op_sources",
    srcs = glob(
        [
            "core/ops/*.cc",
            "data/*.cc",
        ],
        exclude = ["core/ops/*_test.cc"],
    ),
)

filegroup(
    name = "custom_op_headers",
    srcs = glob(
        [
            "core/ops/*.h",
        ],
    ),
)

py_library(
    name = "constants",
    srcs = [
        "python/constants.py",
    ],
    srcs_version = "PY2AND3",
)

tf_custom_op_library(
    name = "data/_data_ops.so",
    srcs = [
        "data/sparse_values_to_indices.cc",
        "data/string_to_float_op.cc",
    ],
    deps = [
        ":tree_utils",
    ],
)

py_library(
    name = "data_ops_lib",
    srcs = [
        "data/data_ops.py",
    ],
    data = [
        "data/_data_ops.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
    ],
)

py_library(
    name = "eval_metrics",
    srcs = ["client/eval_metrics.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "client_lib",
    srcs_version = "PY2AND3",
    deps = [
        ":data_ops_lib",
        ":eval_metrics",
        ":tensor_forest_py",
    ],
)

cc_library(
    name = "tree_utils",
    srcs = ["core/ops/tree_utils.cc"],
    hdrs = [
        "core/ops/tree_utils.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf",
    ],
)

tf_custom_op_library(
    name = "python/ops/_inference_ops.so",
    srcs = [
        "core/ops/tree_predictions_op.cc",
    ],
    deps = [":tree_utils"],
)

tf_custom_op_library(
    name = "python/ops/_training_ops.so",
    srcs = [
        "core/ops/best_splits_op.cc",
        "core/ops/count_extremely_random_stats_op.cc",
        "core/ops/finished_nodes_op.cc",
        "core/ops/grow_tree_op.cc",
        "core/ops/sample_inputs_op.cc",
        "core/ops/scatter_add_ndim_op.cc",
        "core/ops/update_fertile_slots_op.cc",
    ],
    deps = [":tree_utils"],
)

py_library(
    name = "ops_lib",
    srcs = [
        "__init__.py",
        "python/ops/inference_ops.py",
        "python/ops/training_ops.py",
    ],
    data = [
        "python/ops/_inference_ops.so",
        "python/ops/_training_ops.so",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "best_splits_op_test",
    size = "small",
    srcs = ["python/kernel_tests/best_splits_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "count_extremely_random_stats_op_test",
    size = "small",
    srcs = ["python/kernel_tests/count_extremely_random_stats_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "grow_tree_op_test",
    size = "small",
    srcs = ["python/kernel_tests/grow_tree_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "finished_nodes_op_test",
    size = "small",
    srcs = ["python/kernel_tests/finished_nodes_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "sample_inputs_op_test",
    size = "small",
    srcs = ["python/kernel_tests/sample_inputs_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "scatter_add_ndim_op_test",
    size = "small",
    srcs = ["python/kernel_tests/scatter_add_ndim_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "tree_predictions_op_test",
    size = "small",
    srcs = ["python/kernel_tests/tree_predictions_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "update_fertile_slots_op_test",
    size = "small",
    srcs = ["python/kernel_tests/update_fertile_slots_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "tensor_forest_py",
    srcs = ["python/tensor_forest.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":ops_lib",
    ],
)

py_test(
    name = "tensor_forest_test",
    size = "small",
    srcs = ["python/tensor_forest_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_forest_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cc_test(
    name = "training_ops_test",
    size = "small",
    srcs = [
        "core/ops/training_ops_test.cc",
        ":custom_op_sources",
    ],
    deps = [
        ":tree_utils",
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
    ],
)
