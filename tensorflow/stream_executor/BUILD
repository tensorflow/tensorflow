licenses(["restricted"])

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_additional_all_protos")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")
load("//tensorflow:tensorflow.bzl", "cc_header_only_library")

STREAM_EXECUTOR_HEADERS = glob([
    "*.h",
    "cuda/*.h",
    "host/*.h",
    "lib/*.h",
    "lib/gtl/*.h",
    "platform/**/*.h",
])

tf_proto_library(
    name = "dnn_proto",
    srcs = ["dnn.proto"],
    cc_api_version = 2,
    default_header = True,
    protodeps = tf_additional_all_protos(),
)

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
    hdrs = STREAM_EXECUTOR_HEADERS,
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":dnn_proto_cc_impl",
        "//tensorflow/core:lib",
        "//tensorflow/core:ptr_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stream_executor",
    hdrs = STREAM_EXECUTOR_HEADERS,
    visibility = ["//visibility:public"],
    deps = [
        ":dnn_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:ptr_util",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
    ] + if_static([":stream_executor_impl"]),
)

cc_header_only_library(
    name = "stream_executor_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":stream_executor",
    ],
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
        "//tensorflow:windows": [],
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
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cudnn",
    ]),
    alwayslink = 1,
)
