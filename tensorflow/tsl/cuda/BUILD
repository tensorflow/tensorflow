# Description:
#   Stubs for dynamically loading CUDA.

load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "cublas_stub",
    srcs = if_cuda_is_configured(["cublas_stub.cc"]),
    textual_hdrs = glob(["cublas_*.inc"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
)

alias(
    name = "cublas_lib",
    actual = select({
        "//tensorflow/tsl:oss": ":cublas_stub",
        "//conditions:default": "@local_config_cuda//cuda:cublas",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas_lt_stub",
    srcs = if_cuda_is_configured(["cublasLt_stub.cc"]),
    textual_hdrs = glob(["cublasLt_*.inc"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
)

alias(
    name = "cublas_lt_lib",
    actual = select({
        "//tensorflow/tsl:oss": ":cublas_lt_stub",
        "//conditions:default": "@local_config_cuda//cuda:cublasLt",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_stub",
    srcs = if_cuda_is_configured(["cuda_stub.cc"]),
    textual_hdrs = glob(["cuda_*.inc"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
)

cc_library(
    name = "cudart_stub",
    srcs = select({
        # include dynamic loading implementation only when if_cuda_is_configured and build dynamically
        "//tensorflow/tsl:is_cuda_enabled_and_oss": ["cudart_stub.cc"],
        "//conditions:default": [],
    }),
    textual_hdrs = glob(["cuda_runtime_*.inc"]),
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow/tsl:is_cuda_enabled_and_oss": [
            ":cuda_stub",
            "@local_config_cuda//cuda:cuda_headers",
            "//tensorflow/tsl/platform:dso_loader",
            "//tensorflow/tsl/platform:env",
        ],
        "//conditions:default": [],
    }),
)

filegroup(
    name = "cudnn_wrappers",
    srcs = glob(["cudnn_*.inc"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_stub",
    srcs = if_cuda_is_configured(["cudnn_stub.cc"]),
    textual_hdrs = glob(["cudnn_*.inc"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cudnn_version",
        "@local_config_cuda//cuda:cudnn_header",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
)

alias(
    name = "cudnn_lib",
    actual = select({
        "//tensorflow/tsl:oss": ":cudnn_stub",
        "//conditions:default": "@local_config_cuda//cuda:cudnn",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_version",
    srcs = ["cudnn_version.cc"],
    hdrs = ["cudnn_version.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "cudnn_version_test",
    srcs = ["cudnn_version_test.cc"],
    deps = [
        ":cudnn_version",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "cufft_stub",
    srcs = if_cuda_is_configured(["cufft_stub.cc"]),
    textual_hdrs = glob(["cufft_*.inc"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
)

alias(
    name = "cufft_lib",
    actual = select({
        "//tensorflow/tsl:oss": ":cufft_stub",
        "//conditions:default": "@local_config_cuda//cuda:cufft",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cupti_stub",
    srcs = if_cuda_is_configured(["cupti_stub.cc"]),
    data = if_cuda_is_configured(["@local_config_cuda//cuda:cupti_dsos"]),
    textual_hdrs = ["cupti_10_0.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cupti_headers",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
)

cc_library(
    name = "curand_stub",
    srcs = if_cuda_is_configured(["curand_stub.cc"]),
    textual_hdrs = ["curand_10_0.inc"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
)

alias(
    name = "curand_lib",
    actual = select({
        "//tensorflow/tsl:oss": ":curand_stub",
        "//conditions:default": "@local_config_cuda//cuda:curand",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusolver_stub",
    srcs = if_cuda_is_configured(["cusolver_stub.cc"]),
    textual_hdrs = glob(["cusolver_dense_*.inc"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
)

alias(
    name = "cusolver_lib",
    actual = select({
        "//tensorflow/tsl:oss": ":cusolver_stub",
        "//conditions:default": "@local_config_cuda//cuda:cusolver",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusparse_stub",
    srcs = if_cuda_is_configured(["cusparse_stub.cc"]),
    textual_hdrs = glob(["cusparse_*.inc"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/tsl/platform:dso_loader",
        "//tensorflow/tsl/platform:env",
    ]),
)

alias(
    name = "cusparse_lib",
    actual = select({
        "//tensorflow/tsl:oss": ":cusparse_stub",
        "//conditions:default": "@local_config_cuda//cuda:cusparse",
    }),
    visibility = ["//visibility:public"],
)
