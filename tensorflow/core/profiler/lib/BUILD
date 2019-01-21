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
    name = "eager_profiler",
    srcs = [
        "eager_profiler.cc",
    ],
    hdrs = [
        "eager_profiler.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/contrib/tpu/profiler:trace_events_proto_cc",
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
