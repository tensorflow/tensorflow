# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_kernel_library",
    "tf_gen_op_wrapper_py",
)

tf_custom_op_py_library(
    name = "seq2seq_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    dso = [
        ":python/ops/_beam_search_ops.so",
    ],
    kernels = [
        ":beam_search_ops_kernels",
        ":beam_search_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":beam_search_ops",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:layers_base",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_custom_op_library(
    name = "python/ops/_beam_search_ops.so",
    srcs = [
        "kernels/beam_search_ops.cc",
        "kernels/beam_search_ops.h",
        "ops/beam_search_ops.cc",
    ],
    gpu_srcs = [
        "kernels/beam_search_ops_gpu.cu.cc",
        "kernels/beam_search_ops.h",
    ],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
    ],
)

tf_gen_op_wrapper_py(
    name = "beam_search_ops",
    deps = [":beam_search_ops_op_lib"],
)

tf_gen_op_libs(
    op_lib_names = [
        "beam_search_ops",
    ],
)

tf_kernel_library(
    name = "beam_search_ops_kernels",
    prefix = "kernels/beam_search_ops",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "//third_party/eigen3",
    ],
)

cuda_py_test(
    name = "loss_test",
    size = "medium",
    srcs = ["python/kernel_tests/loss_test.py"],
    additional_deps = [
        ":seq2seq_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
    ],
)

cuda_py_test(
    name = "basic_decoder_test",
    size = "medium",
    srcs = ["python/kernel_tests/basic_decoder_test.py"],
    additional_deps = [
        ":seq2seq_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:rnn",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "beam_search_ops_test",
    size = "medium",
    srcs = ["python/kernel_tests/beam_search_ops_test.py"],
    additional_deps = [
        ":seq2seq_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "decoder_test",
    size = "medium",
    srcs = ["python/kernel_tests/decoder_test.py"],
    additional_deps = [
        ":seq2seq_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:rnn",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "beam_search_decoder_test",
    size = "small",
    srcs = ["python/kernel_tests/beam_search_decoder_test.py"],
    additional_deps = [
        ":seq2seq_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:rnn",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "attention_wrapper_test",
    size = "medium",
    srcs = ["python/kernel_tests/attention_wrapper_test.py"],
    additional_deps = [
        ":seq2seq_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:rnn",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
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
