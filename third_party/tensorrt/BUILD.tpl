# NVIDIA TensorRT
# A high-performance deep learning inference optimizer and runtime.

licenses(["notice"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorrt_headers",
    hdrs = [%{tensorrt_headers}],
    includes = [
        "include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nv_infer",
    srcs = [%{nv_infer}],
    data = [%{nv_infer}],
    includes = [
        "include",
    ],
    copts= cuda_default_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda",
        ":tensorrt_headers",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nv_infer_plugin",
    srcs = [%{nv_infer_plugin}],
    data = [%{nv_infer_plugin}],
    includes = [
        "include",
    ],
    copts= cuda_default_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda",
        ":nv_infer",
        ":tensorrt_headers",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nv_parsers",
    srcs = [%{nv_parsers}],
    data = [%{nv_parsers}],
    includes = [
        "include",
    ],
    copts= cuda_default_copts(),
    deps = [
        ":tensorrt_headers",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

%{tensorrt_genrules}
