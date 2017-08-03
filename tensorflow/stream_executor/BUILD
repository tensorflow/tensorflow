licenses(["restricted"])

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")

cc_library(
    name = "stream_executor_impl",
    srcs = glob(
        [
            "*.cc",
            "host/*.cc",
            "cuda/cuda_platform_id.cc",
            "lib/*.cc",
            "platform/default/*.cc",
        ],
        exclude = [
            "**/*_test.cc",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [":stream_executor"],
    alwayslink = 1,
)

cc_library(
    name = "stream_executor",
    hdrs = glob([
        "*.h",
        "cuda/*.h",
        "host/*.h",
        "lib/*.h",
        "lib/gtl/*.h",
        "platform/**/*.h",
    ]),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//conditions:default": ["-ldl"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cuda_platform",
    srcs = if_cuda_is_configured(
        glob(
            [
                "cuda/*.cc",
            ],
            exclude = ["cuda/cuda_platform_id.cc"],
        ),
    ),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//conditions:default": ["-ldl"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":stream_executor",
        "//tensorflow/core:lib",
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
