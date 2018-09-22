# Description:
#   Wrap AMD (https://github.com/ROCmSoftwarePlatform/rccl) RCCL with tensorflow ops.
#   APIs are meant to change over time.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "gpu_py_test",
    "tf_custom_op_library",
    "tf_custom_op_py_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

tf_custom_op_py_library(
    name = "rccl_py",
    srcs = [
        "__init__.py",
        "python/ops/rccl_ops.py",
    ],
    dso = [
        ":python/ops/_rccl_ops.so" ,
    ],
    kernels = [
        ":rccl_kernels",
        ":rccl_ops_op_lib",
    ],
    deps = [
        ":gen_rccl_ops",
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
    deps = if_rocm_is_configured([
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:core_cpu_lib",
        "@local_config_rocm//rocm:rccl",
    ]),
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_rccl_ops.so",
    srcs = [
        "kernels/rccl_manager.cc",
        "kernels/rccl_manager.h",
        "kernels/rccl_ops.cc",
        "kernels/rccl_rewrite.cc",
        "ops/rccl_ops.cc",
    ],
    deps = if_rocm_is_configured([
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:core_cpu_lib",
        "@local_config_rocm//rocm:rccl",
    ]),
)

tf_gen_op_libs(
    op_lib_names = ["rccl_ops"],
)

tf_gen_op_wrapper_py(
    name = "gen_rccl_ops",
    out = "python/ops/gen_rccl_ops.py",
    deps = [":rccl_ops_op_lib"],
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
