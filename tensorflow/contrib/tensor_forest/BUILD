# TensorFlow code for training random forests.
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//visibility:public",
])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

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
            "kernels/*.cc",
            "ops/*.cc",
        ],
        exclude = [
            "kernels/*_test.cc",
            "kernels/tree_utils.cc",
        ],
    ),
)

filegroup(
    name = "custom_op_headers",
    srcs = glob(
        [
            "kernels/*.h",
        ],
        exclude = [
            "kernels/data_spec.h",
            "kernels/tree_utils.h",
        ],
    ),
)

cc_library(
    name = "all_ops",
    srcs = [":custom_op_sources"],
    hdrs = [":custom_op_headers"],
    deps = [
        ":tree_utils",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf_headers",
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
        "python/ops/data_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_forest_ops_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
    ],
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
        "kernels/best_splits_op.cc",
        "kernels/count_extremely_random_stats_op.cc",
        "kernels/finished_nodes_op.cc",
        "kernels/grow_tree_op.cc",
        "kernels/reinterpret_string_to_float_op.cc",
        "kernels/sample_inputs_op.cc",
        "kernels/scatter_add_ndim_op.cc",
        "kernels/tree_predictions_op.cc",
        "kernels/update_fertile_slots_op.cc",
        "ops/tensor_forest_ops.cc",
    ],
    deps = [":tree_utils"],
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
        "client/__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":data_ops_py",
        ":eval_metrics",
        ":random_forest",
        ":tensor_forest_ops_py",
        ":tensor_forest_py",
    ],
)

tf_kernel_library(
    name = "tensor_forest_kernels",
    srcs = [
        "kernels/best_splits_op.cc",
        "kernels/count_extremely_random_stats_op.cc",
        "kernels/finished_nodes_op.cc",
        "kernels/grow_tree_op.cc",
        "kernels/reinterpret_string_to_float_op.cc",
        "kernels/sample_inputs_op.cc",
        "kernels/scatter_add_ndim_op.cc",
        "kernels/tree_predictions_op.cc",
        "kernels/update_fertile_slots_op.cc",
    ],
    deps = [
        ":tree_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:bounds_check",
    ],
)

tf_custom_op_py_library(
    name = "tensor_forest_ops_py",
    srcs = ["python/ops/tensor_forest_ops.py"],
    dso = ["python/ops/_tensor_forest_ops.so"],
    kernels = [
        ":tensor_forest_kernels",
        ":tensor_forest_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":gen_tensor_forest_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
    ],
)

py_library(
    name = "eval_metrics",
    srcs = ["client/eval_metrics.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/learn:estimator_constants_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "eval_metrics_test",
    size = "small",
    srcs = ["client/eval_metrics_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":eval_metrics",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
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
    srcs = ["kernels/tree_utils.cc"],
    hdrs = [
        "kernels/data_spec.h",
        "kernels/tree_utils.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf_headers",
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
        ":data_ops_py",
        ":tensor_forest_ops_py",
        "//tensorflow/core:protos_all_py",
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
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
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
        ":data_ops_py",
        ":tensor_forest_ops_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "scatter_add_ndim_op_test",
    size = "small",
    srcs = ["python/kernel_tests/scatter_add_ndim_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip_gpu"],
    deps = [
        ":tensor_forest_ops_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
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
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "tensor_forest_test",
    size = "small",
    srcs = ["python/tensor_forest_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_forest_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cc_test(
    name = "tensor_forest_ops_test",
    size = "small",
    srcs = [
        "kernels/tensor_forest_ops_test.cc",
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
    name = "random_forest",
    srcs = ["client/random_forest.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_lib",
        ":data_ops_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
    ],
)

py_test(
    name = "random_forest_test",
    size = "medium",
    srcs = ["client/random_forest_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":random_forest",
        ":tensor_forest_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//third_party/py/numpy",
    ],
)
