package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_additional_all_protos")

tf_cc_binary(
    name = "profiler",
    srcs = ["profiler.cc"],
    deps = [
        ":protos_all_cc",
        "//tensorflow/c:c_api",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
        "//tensorflow/core/profiler/internal:tfprof_stats",
        "//tensorflow/core/profiler/internal:tfprof_utils",
        "//tensorflow/core/profiler/internal/advisor:tfprof_advisor",
        "@linenoise",
    ],
)

cc_library(
    name = "tfprof_options",
    srcs = ["tfprof_options.cc"],
    hdrs = ["tfprof_options.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:protos_all_cc",
    ],
)

tf_proto_library(
    name = "protos_all",
    srcs = glob(["**/*.proto"]),
    cc_api_version = 2,
    java_api_version = 2,
    protodeps = tf_additional_all_protos(),
    visibility = ["//visibility:public"],
)
