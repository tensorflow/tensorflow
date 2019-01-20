# NVIDIA TensorRT
# A high-performance deep learning inference optimizer and runtime.

licenses(["notice"])

exports_files(["LICENSE"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorrt_headers",
    hdrs = [%{tensorrt_headers}],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorrt",
    srcs = %{tensorrt_libs},
    data = %{tensorrt_libs},
    copts= cuda_default_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda",
        ":tensorrt_headers",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

%{copy_rules}

