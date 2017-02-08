# Description:
#   Wrap NVIDIA (https://github.com/NVIDIA/nccl) NCCL with tensorflow ops.
#   APIs are meant to change over time.
package(
    default_visibility = ["//visibility:private"],
    features = ["-parse_headers"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_cc_test",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

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

py_library(
    name = "nccl_py",
    srcs = [
        "__init__.py",
        "python/ops/nccl_ops.py",
    ],
    data = [
        ":python/ops/_nccl_ops.so",
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
    tags = [
        "manual",
        "requires_cudnn5",
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
    tags = ["manual"],  # Disabled until errors finding nvmlShutdown are found.
    deps = if_cuda(
        [
            "@nccl_archive//:nccl",
            "//tensorflow/core",
            "//tensorflow/core:cuda",
        ],
        [],
    ) + [
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
    visibility = ["//tensorflow:__subpackages__"],
)
