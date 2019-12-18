load("//tensorflow:tensorflow.bzl", "tf_cuda_library")

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "profiler_session",
    srcs = ["profiler_session.cc"],
    hdrs = ["profiler_session.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":profiler_utils",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/internal:profiler_factory",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
    ] + select({
        "//tensorflow:android": [],
        "//conditions:default": [
            "//tensorflow/core/profiler/convert:run_metadata_to_trace_events",
            "//tensorflow/core/platform",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
        ],
    }),
)

tf_cuda_library(
    name = "profiler_lib",
    cuda_deps = [
        "//tensorflow/core/profiler/internal/gpu:device_tracer",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/profiler/internal/cpu:host_tracer",
    ],
    alwayslink = True,
)

filegroup(
    name = "pywrap_eager_hdrs",
    srcs = [
        "profiler_session.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

cc_library(
    name = "traceme",
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "//tensorflow/core/profiler/internal:traceme_recorder",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "scoped_annotation",
    hdrs = ["scoped_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "//tensorflow/core/profiler/internal:annotation_stack",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "profiler_utils",
    srcs = ["profiler_utils.cc"],
    hdrs = ["profiler_utils.h"],
    visibility = ["//tensorflow/core/profiler:internal"],
)

filegroup(
    name = "mobile_srcs",
    srcs = [
        "profiler_session.cc",
        "profiler_session.h",
        "scoped_annotation.h",
        "traceme.h",
    ],
    visibility = ["//visibility:public"],
)
