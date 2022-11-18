load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "tf_grpc_cc_dependencies")
load(
    "//tensorflow/core/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "capture_profile",
    hdrs = ["capture_profile.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/compiler/xla/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_proto_cc",
        "//tensorflow/tsl/profiler/rpc/client:capture_profile",
    ],
)

cc_library(
    name = "save_profile",
    hdrs = ["save_profile.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_proto_cc",
        "//tensorflow/tsl/profiler/rpc/client:save_profile",
    ],
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_client_for_pybind",
    actual = ":profiler_client",
)

cc_library(
    name = "profiler_client",
    hdrs = ["profiler_client.h"],
    visibility = ["//tensorflow/compiler/xla:__subpackages__"],
    deps = [
        ":profiler_client_impl",
        "//tensorflow/core:lib",
        "//tensorflow/tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "//tensorflow/tsl/profiler/rpc/client:profiler_client",
        "//tensorflow/tsl/profiler/rpc/client:profiler_client_impl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

# Linked to pywrap_tensorflow to avoid ODR violation due to tf_grpc_cc_dependencies().
cc_library(
    name = "profiler_client_impl",
    srcs = [
        "profiler_client.h",
    ],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/python:__pkg__"],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "//tensorflow/tsl/profiler/rpc/client:profiler_client",
        "//tensorflow/tsl/profiler/rpc/client:profiler_client_impl",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = True,
)

cc_library(
    name = "remote_profiler_session_manager",
    hdrs = ["remote_profiler_session_manager.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":profiler_client_for_pybind",
        "//tensorflow/core:lib",
        "//tensorflow/tsl/profiler/rpc/client:remote_profiler_session_manager",
        "@com_google_absl//absl/strings",
    ],
)
