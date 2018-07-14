licenses(["restricted"])

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")

STREAM_EXECUTOR_HEADERS = glob([
    "*.h",
    "cuda/*.h",
    "rocm/*.h",
    "host/*.h",
    "lib/*.h",
    "lib/gtl/*.h",
    "platform/**/*.h",
])

cc_library(
    name = "stream_executor_impl",
    srcs = glob(
        [
            "*.cc",
            "host/*.cc",
            "cuda/cuda_platform_id.cc",
            "rocm/rocm_platform_id.cc",
            "lib/*.cc",
            "platform/default/*.cc",
        ],
        exclude = [
            "**/*_test.cc",
        ],
    ),
    hdrs = STREAM_EXECUTOR_HEADERS,
    copts = ["-g", "-O0",],
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//conditions:default": ["-ldl"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:lib",
        "//tensorflow/core:ptr_util",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stream_executor",
    hdrs = STREAM_EXECUTOR_HEADERS,
    visibility = ["//visibility:public"],
    copts = ["-g", "-O0",],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:ptr_util",
        "//tensorflow/compiler/xla:statusor",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_rocm//rocm:rocm_headers",
    ] + if_static([":stream_executor_impl"]),
)

cc_library(
    name = "cuda_platform",
    srcs = if_cuda_is_configured(
        glob(
            [
                "cuda/*.cc",
            ],
            exclude = [
                "cuda/*_test.cc",
                "cuda/cuda_platform_id.cc",
            ],
        ),
    ),
    copts = select({
        "//tensorflow:windows": ["/DNOGDI"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//conditions:default": ["-ldl"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":stream_executor",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:ops_util",
        "@local_config_cuda//cuda:cuda_headers",
    ] + if_cuda_is_configured([
        "//tensorflow/core:cuda",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:curand",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "rocm_platform",
    srcs = if_rocm_is_configured(
        glob(
            [
                "rocm/*.cc",
            ],
            exclude = ["rocm/rocm_platform_id.cc"],
        ),
    ),
    copts = ["-g", "-O0",],
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//conditions:default": ["-ldl"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":stream_executor",
        "//tensorflow/core:lib",
        "@local_config_rocm//rocm:rocm_headers",
    ] + if_rocm_is_configured([
        "//tensorflow/core:rocm",
        "@local_config_rocm//rocm:hiprand",
        "@local_config_rocm//rocm:rocfft",
        "@local_config_rocm//rocm:rocblas",
        "@local_config_rocm//rocm:miopen",
    ]),
    alwayslink = 1,
)
