load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependency")
load("//tensorflow:tensorflow.bzl", "tf_pybind_cc_library_wrapper")  # buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

# For platform specific build config
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_profiler_client_deps",
    "tf_protos_profiler_service",
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
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        tf_grpc_cc_dependency(),
    ],
    alwayslink = True,
)

tf_pybind_cc_library_wrapper(
    name = "profiler_client_headers",
    visibility = ["//tensorflow/python/profiler/internal:__pkg__"],
    deps = [":profiler_client_impl"],
)

tf_cc_test(
    name = "profiler_client_test",
    srcs = [
        "profiler_client_test.cc",
        "profiler_client_test_util.h",
    ],
    tags = ["external"],  # So that test suite reruns unconditionally.
    deps = [
        ":profiler_client_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/rpc:profiler_server_impl",
    ] + tf_protos_profiler_service(),
)

cc_library(
    name = "remote_profiler_session_manager",
    srcs = ["remote_profiler_session_manager.cc"],
    hdrs = ["remote_profiler_session_manager.h"],
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        ":profiler_client_impl",
        ":save_profile",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/utils:time_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "remote_profiler_session_manager_test",
    srcs = [
        "profiler_client_test_util.h",
        "remote_profiler_session_manager_test.cc",
    ],
    tags = ["external"],  # So that test suite reruns unconditionally.
    deps = [
        ":remote_profiler_session_manager",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/rpc:profiler_server_impl",
    ] + tf_protos_profiler_service(),
)
