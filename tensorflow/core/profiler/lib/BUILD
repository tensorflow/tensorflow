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
        "@com_google_absl//absl/strings",
    ] + select({
        "//tensorflow:android": [],
        "//conditions:default": [
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
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

cc_library(
    name = "traceme",
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
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
        "profiler_utils.cc",
        "profiler_utils.h",
        "scoped_annotation.h",
        "traceme.h",
    ],
    visibility = ["//visibility:public"],
)
