# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_cc_test",
    "tf_py_test",
    "tf_gen_op_libs",
    "tf_kernel_library",
    "tf_gen_op_wrapper_py",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

cc_library(
    name = "all_ops",
    deps = [
        ":gru_ops_op_lib",
        ":lstm_ops_op_lib",
    ],
)

cc_library(
    name = "all_kernels",
    deps = [
        ":gru_ops_kernels",
        ":lstm_ops_kernels",
    ],
)

tf_custom_op_py_library(
    name = "rnn_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]) + [
        "python/tools/checkpoint_convert.py",
    ],
    dso = [
        ":python/ops/_gru_ops.so",
        ":python/ops/_lstm_ops.so",
    ],
    kernels = [
        ":all_ops",
        ":all_kernels",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":benchmarking",
        ":gru_ops",
        ":lstm_ops",
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
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_tests(
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
    xla_enabled = True,
)

cuda_py_tests(
    name = "core_rnn_cell_test",
    size = "small",
    srcs = ["python/kernel_tests/core_rnn_cell_test.py"],
    additional_deps = [
        ":rnn_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_tests(
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

cuda_py_tests(
    name = "core_rnn_test",
    size = "large",
    srcs = ["python/kernel_tests/core_rnn_test.py"],
    additional_deps = [
        ":rnn_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:rnn",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
    ],
    shard_count = 10,
)

tf_py_test(
    name = "fused_rnn_cell_test",
    size = "small",
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
    tags = [
        "manual",
        "notap",
    ],
)

cuda_py_tests(
    name = "lstm_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/lstm_ops_test.py"],
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
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

tf_custom_op_library(
    name = "python/ops/_lstm_ops.so",
    srcs = [
        "kernels/blas_gemm.cc",
        "kernels/blas_gemm.h",
        "kernels/lstm_ops.cc",
        "kernels/lstm_ops.h",
        "ops/lstm_ops.cc",
    ],
    gpu_srcs = [
        "kernels/blas_gemm.h",
        "kernels/lstm_ops_gpu.cu.cc",
        "kernels/lstm_ops.h",
    ],
    deps = ["//tensorflow/core/kernels:eigen_helpers"],
)

tf_gen_op_wrapper_py(
    name = "lstm_ops",
    deps = [":lstm_ops_op_lib"],
)

tf_custom_op_library(
    name = "python/ops/_gru_ops.so",
    srcs = [
        "kernels/blas_gemm.cc",
        "kernels/blas_gemm.h",
        "kernels/gru_ops.cc",
        "kernels/gru_ops.h",
        "ops/gru_ops.cc",
    ],
    gpu_srcs = [
        "kernels/blas_gemm.h",
        "kernels/gru_ops_gpu.cu.cc",
        "kernels/gru_ops.h",
    ],
    deps = ["//tensorflow/core/kernels:eigen_helpers"],
)

tf_gen_op_wrapper_py(
    name = "gru_ops",
    deps = [":gru_ops_op_lib"],
)

cuda_py_tests(
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
    tags = ["no_oss"],
)

tf_cc_test(
    name = "ops/gru_ops_test",
    size = "small",
    srcs = ["ops/gru_ops_test.cc"],
    data = [":python/ops/_gru_ops.so"],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    # linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "ops/lstm_ops_test",
    size = "small",
    srcs = ["ops/lstm_ops_test.cc"],
    data = [":python/ops/_lstm_ops.so"],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    # linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "lstm_ops",
        "gru_ops",
    ],
)

tf_kernel_library(
    name = "gru_ops_kernels",
    srcs = [
        "kernels/blas_gemm.cc",
        "kernels/blas_gemm.h",
    ],
    gpu_srcs = [
        "kernels/blas_gemm.h",
    ],
    prefix = "kernels/gru_ops",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "lstm_ops_kernels",
    srcs = [
        "kernels/blas_gemm.cc",
        "kernels/blas_gemm.h",
    ],
    gpu_srcs = [
        "kernels/blas_gemm.h",
    ],
    prefix = "kernels/lstm_ops",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "//third_party/eigen3",
    ],
)

py_binary(
    name = "checkpoint_convert",
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
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":checkpoint_convert",
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
