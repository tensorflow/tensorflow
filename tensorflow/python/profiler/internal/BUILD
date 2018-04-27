package(default_visibility = ["//tensorflow/python/profiler:__subpackages__"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "gpu_py_test")
load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")

py_library(
    name = "flops_registry",
    srcs = ["flops_registry.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:graph_util",
    ],
)

py_library(
    name = "model_analyzer_testlib",
    srcs = ["model_analyzer_testlib.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "print_model_analysis_test",
    srcs = ["print_model_analysis_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:variable_scope",
    ],
)

gpu_py_test(
    name = "run_metadata_test",
    srcs = ["run_metadata_test.py"],
    additional_deps = [
        ":model_analyzer_testlib",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/profiler:model_analyzer",
        "//tensorflow/python:random_ops",
    ],
    tags = [
        "no_pip",
    ],
)
