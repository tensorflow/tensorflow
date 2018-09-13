# Description:
#   Wrap NVIDIA (https://github.com/NVIDIA/nccl) NCCL with tensorflow ops.
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
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "if_not_windows_cuda")

tf_custom_op_library(
    name = "python/ops/_nccl_ops.so",
    srcs = [
        "ops/nccl_ops.cc",
    ],
    gpu_srcs = if_not_windows_cuda([
        "kernels/nccl_manager.cc",
        "kernels/nccl_manager.h",
        "kernels/nccl_ops.cc",
    ]),
    deps = [] + if_cuda([
        "@local_config_nccl//:nccl",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:protos_all_proto_text",
    ]),
)

tf_cuda_cc_test(
    name = "nccl_manager_test",
    size = "medium",
    srcs = if_cuda(
        [
            "kernels/nccl_manager.cc",
            "kernels/nccl_manager.h",
            "kernels/nccl_manager_test.cc",
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
        if_cuda([
            "@local_config_nccl//:nccl",
            "//tensorflow/core:cuda",
            "//tensorflow/core:test",
            "//tensorflow/core:test_main",
            "//tensorflow/core:testlib",
        ]),
)

tf_kernel_library(
    name = "nccl_kernels",
    srcs = if_cuda([
        "kernels/nccl_manager.cc",
        "kernels/nccl_manager.h",
        "kernels/nccl_ops.cc",
        "kernels/nccl_rewrite.cc",
    ]),
    deps = if_cuda([
        "@local_config_nccl//:nccl",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
    ]),
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["nccl_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "nccl_ops",
    deps = [":nccl_ops_op_lib"],
)

# Test only nccl ops lib without dso to test behavior when NCCL lib is not
# installed. See nccl_dependency_test for more details.
#
# Users should use the public nccl_py lib that also adds the dso.
tf_custom_op_py_library(
    name = "nccl_ops_lib_without_dso",
    srcs = [
        "__init__.py",
        "python/ops/nccl_ops.py",
    ],
    kernels = if_cuda([":nccl_kernels"]) + [
        ":nccl_ops_op_lib",
    ],
    deps = [
        ":nccl_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
    ],
)

tf_custom_op_py_library(
    name = "nccl_py",
    dso = [":python/ops/_nccl_ops.so"],
    visibility = ["//visibility:public"],
    deps = [
        ":nccl_ops_lib_without_dso",
    ],
)

cuda_py_test(
    name = "nccl_ops_test",
    size = "small",
    srcs = ["python/ops/nccl_ops_test.py"],
    additional_deps = [
        ":nccl_py",
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

cuda_py_test(
    name = "nccl_dependency_test",
    size = "small",
    srcs = ["python/ops/nccl_dependency_test.py"],
    additional_deps = [
        ":nccl_ops_lib_without_dso",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
    ],
    # Disable this test internally as static linking is used internally and only
    # run for OSS to verify that NCCL is an optional dynamic dependency.
    tags = [
        "manual",
        "noguitar",
        "notap",
    ],
)
