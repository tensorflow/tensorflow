# Description: Batching scheduling library.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "batch_scheduler_hdrs",
    hdrs = ["batch_scheduler.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "batch_scheduler",
    hdrs = ["batch_scheduler.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "batch_scheduler_test",
    srcs = ["batch_scheduler_test.cc"],
    deps = [
        ":batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "shared_batch_scheduler_hdrs",
    hdrs = ["shared_batch_scheduler.h"],
    deps = [
        ":batch_scheduler_hdrs",
        "//tensorflow/contrib/batching/util:periodic_function_dynamic",
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "shared_batch_scheduler",
    hdrs = ["shared_batch_scheduler.h"],
    deps = [
        ":batch_scheduler",
        "//tensorflow/contrib/batching/util:periodic_function_dynamic",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "shared_batch_scheduler_test",
    srcs = ["shared_batch_scheduler_test.cc"],
    deps = [
        ":shared_batch_scheduler",
        "//tensorflow/contrib/batching/test_util:fake_clock_env",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "basic_batch_scheduler",
    hdrs = ["basic_batch_scheduler.h"],
    deps = [
        ":shared_batch_scheduler",
    ],
)

cc_test(
    name = "basic_batch_scheduler_test",
    srcs = ["basic_batch_scheduler_test.cc"],
    deps = [
        ":basic_batch_scheduler",
        ":batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "basic_batch_scheduler_benchmark",
    srcs = ["basic_batch_scheduler_benchmark.cc"],
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":basic_batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
    ],
)

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "python/ops/_batch_ops.so",
    srcs = ["ops/batch_ops.cc"],
    deps = [
        "//tensorflow/contrib/batching/kernels:batch_kernels",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["batch_ops"],
)

tf_gen_op_wrapper_py(
    name = "batch_ops",
    deps = [":batch_ops_op_lib"],
)

tf_kernel_library(
    name = "batch_ops_kernels",
    deps = [
        "//tensorflow/contrib/batching/kernels:batch_kernels",
        "//tensorflow/contrib/batching/util:periodic_function",
        "//tensorflow/core/kernels:concat_lib",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:split_lib",
    ],
    alwayslink = 1,
)

tf_custom_op_py_library(
    name = "batch_py",
    srcs = glob(["python/ops/*.py"]) + ["__init__.py"],
    dso = [":python/ops/_batch_ops.so"],
    kernels = [
        ":batch_ops_kernels",
        ":batch_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":batch_ops",
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
    name = "batch_ops_test",
    size = "small",
    srcs = ["python/ops/batch_ops_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_pip",
        "nomac",
    ],
    deps = [
        ":batch_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:gradients",
        "//tensorflow/python:script_ops",
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
