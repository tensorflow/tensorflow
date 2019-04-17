package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
)

tf_cuda_library(
    name = "profiler_session",
    srcs = [
        "profiler_session.cc",
    ],
    hdrs = [
        "profiler_session.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/profiler/internal/gpu:device_tracer",
        "//tensorflow/core/profiler/internal/cpu:host_tracer",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler:protos_all_cc",
    ] + select({
        "//tensorflow:android": [],
        "//conditions:default": [
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:session_options",
            "//tensorflow/core:device_tracer",
        ],
    }),
)

tf_cuda_library(
    name = "profiler_graph_lib",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/profiler/internal/cpu:host_tracer",
        "//tensorflow/core/profiler/internal/gpu:device_tracer",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "profiler_eager_lib",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/profiler/internal/cpu:host_tracer",
        "//tensorflow/core/profiler/internal/gpu:device_tracer",
        "//tensorflow/core/profiler/internal/runtime:eager_profiler",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "traceme",
    srcs = ["traceme.cc"],
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/internal:traceme_recorder",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

filegroup(
    name = "mobile_srcs",
    srcs = [
        "profiler_session.cc",
        "profiler_session.h",
    ],
    visibility = ["//visibility:public"],
)
