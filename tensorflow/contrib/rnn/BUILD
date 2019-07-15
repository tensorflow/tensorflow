# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_py_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "rnn_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]) + [
        "python/tools/checkpoint_convert.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":benchmarking",
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:rnn_ops_gen",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "rnn_cell_test",
    size = "medium",
    srcs = ["python/kernel_tests/rnn_cell_test.py"],
    additional_deps = [
        ":rnn_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = ["optonly"],
    xla_enabled = True,
)

cuda_py_test(
    name = "rnn_test",
    size = "medium",
    srcs = ["python/kernel_tests/rnn_test.py"],
    additional_deps = [
        ":rnn_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = [
        "optonly",
    ],
)

tf_py_test(
    name = "fused_rnn_cell_test",
    size = "medium",
    srcs = ["python/kernel_tests/fused_rnn_cell_test.py"],
    additional_deps = [
        ":rnn_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "lstm_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/lstm_ops_test.py"],
    additional_deps = [
        ":rnn_py",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:rnn",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = ["noasan"],
)

cuda_py_test(
    name = "gru_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/gru_ops_test.py"],
    additional_deps = [
        ":rnn_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:rnn",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = [
        "no_oss",
        "noasan",
    ],
)

py_binary(
    name = "checkpoint_convert",
    srcs = ["python/tools/checkpoint_convert.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [":checkpoint_convert_lib"],
)

py_library(
    name = "checkpoint_convert_lib",
    srcs = ["python/tools/checkpoint_convert.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "checkpoint_convert_test",
    size = "small",
    srcs = ["python/tools/checkpoint_convert_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":checkpoint_convert_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "benchmarking",
    srcs = ["python/kernel_tests/benchmarking.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:framework_ops"],
)

# These are compatibility re-exports. Remove once there are
# no usages left.

USE_CORE_TF = "contrib/rnn kernels and ops are now part of core TensorFlow"

cc_library(
    name = "all_ops",
    deprecation = USE_CORE_TF,
    deps = [
        "//tensorflow/core:rnn_ops_op_lib",
    ],
)

cc_library(
    name = "all_kernels",
    deprecation = USE_CORE_TF,
    deps = [
        ":gru_ops_kernels",
        ":lstm_ops_kernels",
    ],
)

cc_library(
    name = "gru_ops_op_lib",
    deprecation = USE_CORE_TF,
    deps = [
        "//tensorflow/core:rnn_ops_op_lib",
    ],
)

cc_library(
    name = "lstm_ops_op_lib",
    deprecation = USE_CORE_TF,
    deps = [
        "//tensorflow/core:rnn_ops_op_lib",
    ],
)

tf_kernel_library(
    name = "gru_ops_kernels",
    deprecation = USE_CORE_TF,
    deps = [
        "//tensorflow/core/kernels/rnn:gru_ops",
    ],
)

tf_kernel_library(
    name = "lstm_ops_kernels",
    deprecation = USE_CORE_TF,
    deps = [
        "//tensorflow/core/kernels/rnn:lstm_ops",
    ],
)
