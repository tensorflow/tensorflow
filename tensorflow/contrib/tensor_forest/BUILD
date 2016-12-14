# TensorFlow code for training random forests.
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//visibility:public",
])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
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
            "ops/*.cc",
        ],
        exclude = [
            "core/ops/*_test.cc",
            "data/*_test.cc",
        ],
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

cc_library(
    name = "all_ops",
    srcs = [":custom_op_sources"],
    hdrs = [":custom_op_headers"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf",
    ],
    alwayslink = 1,
)

py_library(
    name = "constants",
    srcs = [
        "python/constants.py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "data_ops_py",
    srcs = [
        "data/data_ops.py",
    ],
    srcs_version = "PY2AND3",
)

tf_gen_op_libs(
    op_lib_names = ["tensor_forest_ops"],
)

tf_gen_op_wrapper_py(
    name = "gen_tensor_forest_ops",
    out = "python/ops/gen_tensor_forest_ops.py",
    deps = [":tensor_forest_ops_op_lib"],
)

tf_custom_op_library(
    name = "python/ops/_tensor_forest_ops.so",
    srcs = [
        "core/ops/best_splits_op.cc",
        "core/ops/count_extremely_random_stats_op.cc",
        "core/ops/finished_nodes_op.cc",
        "core/ops/grow_tree_op.cc",
        "core/ops/sample_inputs_op.cc",
        "core/ops/scatter_add_ndim_op.cc",
        "core/ops/topn_ops.cc",
        "core/ops/tree_predictions_op.cc",
        "core/ops/update_fertile_slots_op.cc",
        "data/reinterpret_string_to_float_op.cc",
        "ops/tensor_forest_ops.cc",
    ],
    deps = [":tree_utils"],
)

py_library(
    name = "tensor_forest_ops_py",
    srcs = [
        "__init__.py",
        "python/ops/tensor_forest_ops.py",
    ],
    data = [
        ":python/ops/_tensor_forest_ops.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_tensor_forest_ops",
    ],
)

py_library(
    name = "eval_metrics",
    srcs = ["client/eval_metrics.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "eval_metrics_test",
    size = "small",
    srcs = ["client/eval_metrics_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":eval_metrics",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "client_lib",
    srcs_version = "PY2AND3",
    deps = [
        ":eval_metrics",
        ":tensor_forest_ops_py",
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

py_test(
    name = "best_splits_op_test",
    size = "small",
    srcs = ["python/kernel_tests/best_splits_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_forest_ops_py",
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
        ":tensor_forest_ops_py",
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
        ":tensor_forest_ops_py",
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
        ":tensor_forest_ops_py",
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
        ":tensor_forest_ops_py",
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
        ":tensor_forest_ops_py",
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
        ":tensor_forest_ops_py",
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
        ":tensor_forest_ops_py",
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
        ":data_ops_py",
        ":tensor_forest_ops_py",
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
    name = "tensor_forest_ops_test",
    size = "small",
    srcs = [
        "core/ops/tensor_forest_ops_test.cc",
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

py_library(
    name = "topn_py",
    srcs = ["python/topn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":tensor_forest_ops_py",
    ],
)

py_test(
    name = "topn_test",
    size = "small",
    srcs = ["python/topn_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":topn_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)
