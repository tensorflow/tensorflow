package(default_visibility = ["//tensorflow/contrib/tfprof/python/tools/tfprof:__subpackages__"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")

py_library(
    name = "model_analyzer_testlib",
    srcs = ["model_analyzer_testlib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/contrib/tfprof/python/tools/tfprof:model_analyzer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:rnn",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

tf_py_wrap_cc(
    name = "pywrap_tensorflow_print_model_analysis_lib",
    srcs = ["pywrap_tensorflow_print_model_analysis.i"],
    swig_includes = [
        "//tensorflow/python:lib/core/strings.i",
        "//tensorflow/python:platform/base.i",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/tools/tfprof/internal:print_model_analysis_hdr",
        "//util/python:python_headers",
    ],
)

py_test(
    name = "print_model_analysis_test",
    srcs = ["print_model_analysis_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow_print_model_analysis_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
        "//tensorflow/tools/tfprof:protos_all_py",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

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
