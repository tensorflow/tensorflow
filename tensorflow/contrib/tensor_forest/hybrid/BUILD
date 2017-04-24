# TensorFlow code for training hybrid neural network / decision tree models.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//visibility:public",
])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
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
            "core/ops/*.cc",
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

cc_library(
    name = "all_kernels",
    srcs = [
        "core/ops/hard_routing_function_op.cc",
        "core/ops/k_feature_gradient_op.cc",
        "core/ops/k_feature_routing_function_op.cc",
        "core/ops/routing_function_op.cc",
        "core/ops/routing_gradient_op.cc",
        "core/ops/stochastic_hard_routing_function_op.cc",
        "core/ops/stochastic_hard_routing_gradient_op.cc",
        "core/ops/unpack_path_op.cc",
    ],
    deps = [
        ":utils",
        "//tensorflow/contrib/tensor_forest:tree_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "training_ops",
    deps = [":all_kernels"],
)

tf_custom_op_library(
    name = "python/ops/_training_ops.so",
    srcs = [
        "core/ops/hard_routing_function_op.cc",
        "core/ops/k_feature_gradient_op.cc",
        "core/ops/k_feature_routing_function_op.cc",
        "core/ops/routing_function_op.cc",
        "core/ops/routing_gradient_op.cc",
        "core/ops/stochastic_hard_routing_function_op.cc",
        "core/ops/stochastic_hard_routing_gradient_op.cc",
        "core/ops/unpack_path_op.cc",
    ],
    deps = [
        ":utils",
        "//tensorflow/contrib/tensor_forest:tree_utils",
    ],
)

cc_library(
    name = "utils",
    srcs = ["core/ops/utils.cc"],
    hdrs = [
        "core/ops/utils.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf//:protobuf_headers",
    ],
)

tf_custom_op_py_library(
    name = "ops_lib",
    srcs = [
        "__init__.py",
        "python/ops/training_ops.py",
    ],
    dso = [
        "python/ops/_training_ops.so",
    ],
    kernels = [
        ":all_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":training_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "all_layers",
    deps = [
        ":decisions_to_data_layer",
        ":fully_connected_layer",
    ],
)

py_library(
    name = "hybrid_layer",
    srcs = [
        "python/hybrid_layer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
    ],
)

py_test(
    name = "hybrid_layer_test",
    size = "small",
    srcs = [
        "python/hybrid_layer_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":fully_connected_layer",
        ":hybrid_layer",
        ":hybrid_model",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "hybrid_model",
    srcs = [
        "python/hybrid_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "fully_connected_layer",
    srcs = [
        "python/layers/fully_connected.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":hybrid_layer",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

py_test(
    name = "routing_function_op_test",
    size = "small",
    srcs = ["python/kernel_tests/routing_function_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":ops_lib",
        ":training_ops",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "k_feature_routing_function_op_test",
    size = "small",
    srcs = ["python/kernel_tests/k_feature_routing_function_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":ops_lib",
        ":training_ops",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "decisions_to_data_layer",
    srcs = [
        "python/layers/decisions_to_data.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":hybrid_layer",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "decisions_to_data_test",
    srcs = [
        "python/layers/decisions_to_data_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":decisions_to_data_layer",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "decisions_to_data_then_nn",
    srcs = [
        "python/models/decisions_to_data_then_nn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":decisions_to_data_layer",
        ":fully_connected_layer",
        ":hybrid_model",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "hard_decisions_to_data_then_nn",
    srcs = [
        "python/models/hard_decisions_to_data_then_nn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":decisions_to_data_layer",
        ":fully_connected_layer",
        ":hybrid_model",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "decisions_to_data_then_nn_test",
    size = "small",
    srcs = ["python/models/decisions_to_data_then_nn_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":decisions_to_data_then_nn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "k_feature_decisions_to_data_then_nn",
    srcs = [
        "python/models/k_feature_decisions_to_data_then_nn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":decisions_to_data_layer",
        ":fully_connected_layer",
        ":hybrid_model",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "k_feature_decisions_to_data_then_nn_test",
    size = "small",
    srcs = ["python/models/k_feature_decisions_to_data_then_nn_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":k_feature_decisions_to_data_then_nn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "forest_to_data_then_nn",
    srcs = [
        "python/models/forest_to_data_then_nn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":decisions_to_data_layer",
        ":fully_connected_layer",
        ":hybrid_model",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "forest_to_data_then_nn_test",
    size = "small",
    srcs = ["python/models/forest_to_data_then_nn_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":forest_to_data_then_nn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "nn",
    srcs = [
        "python/models/nn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":fully_connected_layer",
        ":hybrid_model",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "stochastic_hard_decisions_to_data_then_nn",
    srcs = [
        "python/models/stochastic_hard_decisions_to_data_then_nn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":decisions_to_data_layer",
        ":fully_connected_layer",
        ":hard_decisions_to_data_then_nn",
        ":hybrid_model",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "stochastic_soft_decisions_to_data_then_nn",
    srcs = [
        "python/models/stochastic_soft_decisions_to_data_then_nn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":decisions_to_data_layer",
        ":fully_connected_layer",
        ":hard_decisions_to_data_then_nn",
        ":hybrid_model",
        ":ops_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "hybrid_pip",
    deps = [
        ":all_layers",
        ":decisions_to_data_then_nn",
        ":forest_to_data_then_nn",
        ":hard_decisions_to_data_then_nn",
        ":k_feature_decisions_to_data_then_nn",
        ":nn",
        ":stochastic_hard_decisions_to_data_then_nn",
        ":stochastic_soft_decisions_to_data_then_nn",
    ],
)
