load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "pybind_extension")
load("//tensorflow:tensorflow.bzl", "py_test", "tf_binary_additional_data_deps", "tf_binary_additional_srcs")

package(
    default_visibility = ["//tensorflow/python/profiler:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

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
    python_version = "PY3",
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

cuda_py_test(
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
    python_version = "PY3",
    tags = [
        "no_gpu",  # b/138442728
        "no_pip",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
)

pybind_extension(
    name = "_pywrap_traceme",
    srcs = ["traceme_wrapper.cc"] + tf_binary_additional_srcs(),
    data = tf_binary_additional_data_deps(),
    module_name = "_pywrap_traceme",
    visibility = [
        "//perftools/accelerators/xprof/xprofilez/integration_tests:__pkg__",
        "//tensorflow/python/profiler:__subpackages__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/types:optional",
        "@pybind11",
    ],
)

pybind_extension(
    name = "_pywrap_scoped_annotation",
    srcs = ["scoped_annotation_wrapper.cc"] + tf_binary_additional_srcs(),
    data = tf_binary_additional_data_deps(),
    module_name = "_pywrap_scoped_annotation",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "@com_google_absl//absl/types:optional",
        "@pybind11",
    ],
)
