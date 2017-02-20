# Description:
#   Contains ops to build an input pipeline for tensorflow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
    "tf_custom_op_library",
    "tf_cc_tests",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

tf_custom_op_library(
    # TODO(sibyl-Mooth6ku,ptucker): Understand why 'python/ops/_' is needed and fix it.
    name = "python/ops/_input_pipeline_ops.so",
    srcs = [
        "ops/input_pipeline_ops.cc",
    ],
    deps = [
        "//tensorflow/contrib/input_pipeline/kernels:input_pipeline_kernels",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["input_pipeline_ops"],
)

tf_gen_op_wrapper_py(
    name = "input_pipeline_ops",
    deps = [":input_pipeline_ops_op_lib"],
)

tf_kernel_library(
    name = "input_pipeline_ops_kernels",
    deps = [
        "//tensorflow/contrib/input_pipeline/kernels:input_pipeline_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

py_library(
    name = "input_pipeline_py",
    srcs = glob(["python/ops/*.py"]) + ["__init__.py"],
    data = [":python/ops/_input_pipeline_ops.so"],
    srcs_version = "PY2AND3",
    deps = [
        ":input_pipeline_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "input_pipeline_ops_test",
    size = "small",
    srcs = ["python/ops/input_pipeline_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":input_pipeline_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_cc_tests(
    name = "input_pipeline_ops_test",
    size = "small",
    srcs = [
        "ops/input_pipeline_ops_test.cc",
    ],
    linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":input_pipeline_ops_op_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
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
)
