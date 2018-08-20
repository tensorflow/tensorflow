package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "common",
    srcs = ["python/common.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

py_test(
    name = "common_test",
    size = "small",
    srcs = ["python/common_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "graph_matcher",
    srcs = [
        "python/graph_matcher.py",
    ],
    srcs_version = "PY2AND3",
    deps = [],
)

py_test(
    name = "graph_matcher_test",
    size = "small",
    srcs = ["python/graph_matcher_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_matcher",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "input_to_ops",
    srcs = ["python/input_to_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common",
    ],
)

py_test(
    name = "input_to_ops_test",
    size = "small",
    srcs = ["python/input_to_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":input_to_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "fold_batch_norms",
    srcs = ["python/fold_batch_norms.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common",
        ":graph_matcher",
        ":input_to_ops",
        "//tensorflow/contrib/graph_editor:graph_editor_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "fold_batch_norms_test",
    srcs = ["python/fold_batch_norms_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fold_batch_norms",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "quant_ops",
    srcs = ["python/quant_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "quant_ops_test",
    size = "small",
    srcs = ["python/quant_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quant_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "quantize",
    srcs = ["python/quantize.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_matcher",
        ":input_to_ops",
        ":quant_ops",
        "//tensorflow/contrib/graph_editor:graph_editor_py",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "quantize_test",
    size = "small",
    srcs = ["python/quantize_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quantize",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "quantize_parameterized_test",
    size = "large",
    srcs = ["python/quantize_parameterized_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":fold_batch_norms",
        ":quantize",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "quantize_graph",
    srcs = [
        "__init__.py",
        "python/quantize_graph.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":fold_batch_norms",
        ":quantize",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "quantize_graph_test",
    size = "small",
    srcs = ["python/quantize_graph_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quantize_graph",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)
