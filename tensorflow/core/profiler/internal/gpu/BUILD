package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
)

tf_cuda_library(
    name = "tracer",
    srcs = [
        "tracer.cc",
    ],
    hdrs = [
        "tracer.h",
    ],
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:device_tracer",
        "//tensorflow/core/profiler/internal:profiler_interface",
    ],
)
