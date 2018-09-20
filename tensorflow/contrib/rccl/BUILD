# Description:
#   Wrap AMD (https://github.com/ROCmSoftwarePlatform/rccl) RCCL with tensorflow ops.
#   APIs are meant to change over time.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.bzl", "tf_gpu_cc_test")
load("//tensorflow:tensorflow.bzl", "gpu_py_test")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_library(
    name = "python/ops/_rccl_ops.so",
    srcs = [
        "ops/rccl_ops.cc",
    ],
    gpu_srcs = if_rocm([
        "kernels/rccl_manager.cc",
        "kernels/rccl_manager.h",
        "kernels/rccl_ops.cc",
    ]),
    deps = if_rocm([
        "@local_config_rocm//rocm:rccl",
        "//tensorflow/core:gpu_headers_lib",
    ]),
)

tf_gpu_cc_test(
    name = "rccl_manager_test",
    size = "medium",
    srcs = if_rocm(
        [
            "kernels/rccl_manager.cc",
            "kernels/rccl_manager.h",
            "kernels/rccl_manager_test.cc",
        ],
        [],
    ),
    # Disabled on jenkins until errors finding nvmlShutdown are found.
    tags = [
        "manual",
        "multi_gpu",
        "no_oss",
        "noguitar",
        "notap",
    ],
    deps =
        [
            "//tensorflow/core:rocm",
            "//tensorflow/core:test",
            "//tensorflow/core:test_main",
            "//tensorflow/core:testlib",
            "@local_config_rocm//rocm:rccl",
        ],
)

tf_kernel_library(
    name = "rccl_kernels",
    srcs = [
        "kernels/rccl_manager.cc",
        "kernels/rccl_manager.h",
        "kernels/rccl_ops.cc",
        "kernels/rccl_rewrite.cc",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:stream_executor",
        "@local_config_rocm//rocm:rccl",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["rccl_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "rccl_ops",
    deps = [":rccl_ops_op_lib"],
)

# Test only rccl ops lib without dso to test behavior when rccl lib is not
# installed. See rccl_dependency_test for more details.
#
# Users should use the public rccl_py lib that also adds the dso.
tf_custom_op_py_library(
    name = "rccl_ops_lib_without_dso",
    srcs = [
        "__init__.py",
        "python/ops/rccl_ops.py",
    ],
    kernels = if_rocm([":rccl_kernels"]) + [
        ":rccl_ops_op_lib",
    ],
    deps = [
        ":rccl_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
    ],
)

tf_custom_op_py_library(
    name = "rccl_py",
    dso = [":python/ops/_rccl_ops.so"],
    visibility = ["//visibility:public"],
    deps = [
        ":rccl_ops_lib_without_dso",
    ],
)

gpu_py_test(
    name = "rccl_ops_test",
    size = "small",
    srcs = ["python/ops/rccl_ops_test.py"],
    additional_deps = [
        ":rccl_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    # Disabled on jenkins until errors finding nvmlShutdown are found.
    tags = [
        "manual",
        "multi_gpu",
        "no_oss",
        "noguitar",
        "notap",
    ],
)

gpu_py_test(
    name = "rccl_dependency_test",
    size = "small",
    srcs = ["python/ops/rccl_dependency_test.py"],
    additional_deps = [
        ":rccl_ops_lib_without_dso",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
    ],
    # Disable this test internally as static linking is used internally and only
    # run for OSS to verify that rccl is an optional dynamic dependency.
    tags = [
        "manual",
        "noguitar",
        "notap",
    ],
)
