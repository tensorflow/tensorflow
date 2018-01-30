# TensorFlow code for training random forests.

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "kernels/v4/*",
            "proto/*",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

# ---------------------------------- V2 ops ------------------------------------------#
filegroup(
    name = "v2_op_sources",
    srcs = [
        "kernels/reinterpret_string_to_float_op.cc",
        "kernels/scatter_add_ndim_op.cc",
    ],
)

filegroup(
    name = "v2_op_defs",
    srcs = [
        "ops/tensor_forest_ops.cc",
    ],
)

cc_library(
    name = "v2_ops",
    srcs = [
        ":v2_op_defs",
        ":v2_op_sources",
    ],
    deps = [
        ":tree_utils",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = 1,
)

py_library(
    name = "data_ops_py",
    srcs = ["python/ops/data_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_forest_ops_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
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
        ":v2_op_defs",
        ":v2_op_sources",
    ] + if_static(
        extra_deps = [],
        otherwise = [
            ":libforestprotos.so",
        ],
    ),
    deps = [
        ":tree_utils",
    ],
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
        ":data_ops_py",
        ":eval_metrics",
        ":model_ops_py",
        ":random_forest",
        ":stats_ops_py",
        ":tensor_forest_ops_py",
        ":tensor_forest_py",
    ],
)

tf_kernel_library(
    name = "tensor_forest_kernels",
    srcs = [":v2_op_sources"],
    deps = [
        ":tree_utils",
        "//tensorflow/core:framework_headers_lib",
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

tf_cc_test(
    name = "tensor_forest_ops_test",
    size = "small",
    srcs = [
        "kernels/tensor_forest_ops_test.cc",
        ":v2_op_defs",
        ":v2_op_sources",
    ],
    deps = [
        ":tree_utils",
        "//tensorflow/core",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
    ],
)

# -------------------------------------- V4 ops ------------------------------- #
cc_library(
    name = "tensor_forest_v4_kernels",
    deps = [
        ":model_ops_kernels",
        ":stats_ops_kernels",
    ],
)

cc_library(
    name = "tensor_forest_v4_ops_op_lib",
    deps = [
        ":model_ops_op_lib",
        ":stats_ops_op_lib",
    ],
)

py_library(
    name = "tensor_forest_v4_ops_py",
    srcs_version = "PY2AND3",
    deps = [
        ":model_ops_py",
        ":stats_ops_py",
    ],
)

# Model Ops.
cc_library(
    name = "model_ops_lib",
    srcs = ["kernels/model_ops.cc"],
    deps = [
        "//tensorflow/contrib/tensor_forest:tree_utils",
        "//tensorflow/contrib/tensor_forest/kernels/v4:decision-tree-resource",
        "//tensorflow/contrib/tensor_forest/kernels/v4:input_data",
        "//tensorflow/core:framework_headers_lib",
    ] + if_static(
        extra_deps = [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        ],
        otherwise = [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc_headers_only",
        ],
    ),
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["model_ops"],
)

tf_gen_op_wrapper_py(
    name = "gen_model_ops_py",
    out = "python/ops/gen_model_ops.py",
    deps = [":model_ops_op_lib"],
)

tf_kernel_library(
    name = "model_ops_kernels",
    deps = [
        ":model_ops_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_model_ops.so",
    srcs = [
        "ops/model_ops.cc",
    ] + if_static(
        extra_deps = [],
        otherwise = [
            ":libforestprotos.so",
        ],
    ),
    deps = [":model_ops_lib"],
)

tf_custom_op_py_library(
    name = "model_ops_py",
    srcs = ["python/ops/model_ops.py"],
    dso = ["python/ops/_model_ops.so"],
    kernels = [
        ":model_ops_kernels",
        ":model_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_model_ops_py",
        ":stats_ops_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:resources",
        "//tensorflow/python:training",
    ],
)

tf_cc_test(
    name = "model_ops_test",
    size = "small",
    srcs = [
        "kernels/model_ops_test.cc",
        "ops/model_ops.cc",
    ],
    deps = [
        ":forest_proto_impl",
        ":model_ops_lib",
        "//tensorflow/contrib/tensor_forest/kernels/v4:decision-tree-resource_impl",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Stats Ops.
cc_library(
    name = "stats_ops_lib",
    srcs = ["kernels/stats_ops.cc"],
    deps = [
        "//third_party/eigen3",
        "//tensorflow/contrib/tensor_forest:tree_utils",
        "//tensorflow/contrib/tensor_forest/kernels/v4:decision-tree-resource",
        "//tensorflow/contrib/tensor_forest/kernels/v4:fertile-stats-resource",
        "//tensorflow/contrib/tensor_forest/kernels/v4:input_data",
        "//tensorflow/contrib/tensor_forest/kernels/v4:input_target",
        "//tensorflow/contrib/tensor_forest/kernels/v4:params",
        "//tensorflow/core:framework_headers_lib",
    ] + if_static(
        extra_deps = [
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
        ],
        otherwise = [
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc_headers_only",
        ],
    ),
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["stats_ops"],
)

tf_gen_op_wrapper_py(
    name = "gen_stats_ops_py",
    out = "python/ops/gen_stats_ops.py",
    deps = [":stats_ops_op_lib"],
)

tf_kernel_library(
    name = "stats_ops_kernels",
    deps = [
        ":stats_ops_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_stats_ops.so",
    srcs = [
        "ops/stats_ops.cc",
    ] + if_static(
        extra_deps = [],
        otherwise = [
            ":libforestprotos.so",
        ],
    ),
    deps = [":stats_ops_lib"],
)

tf_custom_op_py_library(
    name = "stats_ops_py",
    srcs = ["python/ops/stats_ops.py"],
    dso = ["python/ops/_stats_ops.so"],
    kernels = [
        ":stats_ops_kernels",
        ":stats_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_stats_ops_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:resources",
        "//tensorflow/python:training",
    ],
)

tf_cc_test(
    name = "stats_ops_test",
    size = "small",
    srcs = [
        "kernels/stats_ops_test.cc",
        "ops/stats_ops.cc",
    ],
    deps = [
        ":forest_proto_impl",
        ":stats_ops_lib",
        "//tensorflow/contrib/tensor_forest/kernels/v4:decision-tree-resource_impl",
        "//tensorflow/core",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
    ],
)

# ---------------------------------- Common libs ------------------------ #
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
        "@protobuf_archive//:protobuf_headers",
    ],
)

cc_library(
    name = "forest_proto_impl",
    deps = [
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc",
        "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
        "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
    ],
)

# Protocol buffer dependencies shared between multiple op shared objects. This
# avoids attempting to register the same protocol buffer multiple times.
tf_cc_shared_object(
    name = "libforestprotos.so",
    # This object does not depend on TensorFlow.
    framework_so = [],
    linkstatic = 1,
    deps = [
        ":forest_proto_impl",
        "//tensorflow/contrib/tensor_forest/kernels/v4:decision-tree-resource_impl",
        "@protobuf_archive//:protobuf",
    ],
)

# --------------------------------- Python -------------------------------- #

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
        ":tensor_forest_v4_ops_py",
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

py_library(
    name = "tensor_forest_py",
    srcs = ["python/tensor_forest.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":data_ops_py",
        ":tensor_forest_ops_py",
        ":tensor_forest_v4_ops_py",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_py",
        "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "@six_archive//:six",
    ],
)

py_test(
    name = "tensor_forest_test",
    size = "small",
    srcs = ["python/tensor_forest_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_forest_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
    ],
)

py_library(
    name = "random_forest",
    srcs = ["client/random_forest.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_lib",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "random_forest_test",
    size = "medium",
    srcs = ["client/random_forest_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
        "nomac",  # b/63258195
        "notsan",
    ],
    deps = [
        ":random_forest",
        ":tensor_forest_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)
