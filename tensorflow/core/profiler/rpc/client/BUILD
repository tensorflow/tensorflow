load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependency")
load("//tensorflow:tensorflow.bzl", "tf_pybind_cc_library_wrapper")  # buildifier: disable=same-origin-load

# For platform specific build config
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_profiler_client_deps",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "capture_profile",
    srcs = ["capture_profile.cc"],
    hdrs = ["capture_profile.h"],
    visibility = [
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [
        ":save_profile",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_analysis_proto_cc",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "//tensorflow/core/profiler/convert:xplane_to_profile_response",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core:lib_internal",
    ] + tf_profiler_client_deps(),
)

cc_library(
    name = "save_profile",
    srcs = ["save_profile.cc"],
    hdrs = ["save_profile.h"],
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "//tensorflow/core/profiler/utils:file_system_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "profiler_client_impl",
    srcs = ["profiler_client.cc"],
    hdrs = ["profiler_client.h"],
    visibility = ["//tensorflow/python:__pkg__"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:profiler_analysis_proto_cc",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        tf_grpc_cc_dependency(),
    ],
    alwayslink = True,
)

tf_pybind_cc_library_wrapper(
    name = "profiler_client_headers",
    visibility = ["//tensorflow/python/profiler/internal:__pkg__"],
    deps = [":profiler_client_impl"],
)
