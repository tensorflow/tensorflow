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

tf_custom_op_library(
    name = "python/ops/_nccl_ops.so",
    srcs = [
        "kernels/nccl_manager.cc",
        "kernels/nccl_manager.h",
        "kernels/nccl_ops.cc",
        "ops/nccl_ops.cc",
    ],
    deps = [
        "//tensorflow/core:gpu_headers_lib",
        "@nccl_archive//:nccl",
    ],
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
        "no_oss",
        "noguitar",  # note: is run manually there
        "notap",
    ],
    deps = if_cuda(
        [
            "@nccl_archive//:nccl",
            "//tensorflow/core:cuda",
        ],
        [],
    ) + [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "nccl_kernels",
    srcs = [
        "kernels/nccl_manager.cc",
        "kernels/nccl_manager.h",
        "kernels/nccl_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "@nccl_archive//:nccl",
    ],
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

tf_custom_op_py_library(
    name = "nccl_py",
    srcs = [
        "__init__.py",
        "python/ops/nccl_ops.py",
    ],
    dso = [
        ":python/ops/_nccl_ops.so",
    ],
    kernels = [
        ":nccl_kernels",
        ":nccl_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":nccl_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:platform",
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
        "no_oss",
        "noguitar",  # note: is run manually there
        "notap",
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
