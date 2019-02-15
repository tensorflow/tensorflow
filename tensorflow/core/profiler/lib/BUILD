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
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/contrib/tpu/profiler:trace_events_proto_cc",
        "//tensorflow/core/profiler/internal/gpu:tracer",
        "//tensorflow/core/profiler/internal/runtime:eager_profiler",
        "//tensorflow/core/profiler/internal:profiler_interface",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite",
        ],
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
