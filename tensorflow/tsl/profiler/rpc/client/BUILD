load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tf_protos_profiler_service",
    "tsl_cc_test",
)
load(
    "//tensorflow/tsl/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/tsl/profiler:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "capture_profile",
    srcs = ["capture_profile.cc"],
    hdrs = ["capture_profile.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/compiler/xla/python:__pkg__",
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [
        ":profiler_client_for_pybind",
        ":remote_profiler_session_manager",
        ":save_profile",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/convert:trace_events_to_json",
        "//tensorflow/tsl/profiler/convert:xplane_to_trace_events",
        "//tensorflow/tsl/profiler/protobuf:profiler_analysis_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "save_profile",
    srcs = ["save_profile.cc"],
    hdrs = ["save_profile.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/tsl/profiler:internal",
        "//tensorflow/tsl/profiler/rpc:__pkg__",
    ],
    deps = [
        "//tensorflow/tsl/lib/io:zlib_compression_options",
        "//tensorflow/tsl/lib/io:zlib_outputbuffer",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/utils:file_system_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_client_for_pybind",
    actual = ":profiler_client",
    visibility = [
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
)

cc_library(
    name = "profiler_client",
    hdrs = ["profiler_client.h"],
    visibility = [
        "//tensorflow/compiler/xla:__subpackages__",
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [
        ":profiler_client_impl",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

# Linked to pywrap_tensorflow to avoid ODR violation due to tf_grpc_cc_dependencies().
cc_library(
    name = "profiler_client_impl",
    srcs = [
        "profiler_client.cc",
        "profiler_client.h",
    ],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "//tensorflow/tsl/protobuf:error_codes_proto_impl_cc",
    ] + tsl_grpc_cc_dependencies(),
    alwayslink = True,
)

cc_library(
    name = "profiler_client_test_util",
    testonly = 1,
    hdrs = ["profiler_client_test_util.h"],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//tensorflow/tsl/profiler/lib:profiler_session",
        "//tensorflow/tsl/profiler/rpc:profiler_server_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
    ] + tf_protos_profiler_service(),
)

tsl_cc_test(
    name = "profiler_client_test",
    srcs = ["profiler_client_test.cc"],
    deps = [
        ":profiler_client",
        ":profiler_client_impl",  # for oss
        ":profiler_client_test_util",
        "@com_google_absl//absl/time",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/lib:profiler_factory_impl",
        "//tensorflow/tsl/profiler/lib:profiler_session_impl",
        "//tensorflow/tsl/profiler/rpc:profiler_server_impl",
        "//tensorflow/tsl/profiler/rpc:profiler_service_impl",
        "//tensorflow/tsl/profiler/utils:time_utils_impl",
    ] + tf_protos_profiler_service(),
)

cc_library(
    name = "remote_profiler_session_manager",
    srcs = ["remote_profiler_session_manager.cc"],
    hdrs = ["remote_profiler_session_manager.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":profiler_client_for_pybind",
        "//tensorflow/tsl/platform:env_time",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/utils:time_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tsl_cc_test(
    name = "remote_profiler_session_manager_test",
    srcs = ["remote_profiler_session_manager_test.cc"],
    deps = [
        ":profiler_client_impl",  # for oss
        ":profiler_client_test_util",
        ":remote_profiler_session_manager",
        "@com_google_absl//absl/time",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tensorflow/tsl/profiler/lib:profiler_factory_impl",
        "//tensorflow/tsl/profiler/lib:profiler_session_impl",
        "//tensorflow/tsl/profiler/rpc:profiler_server_impl",
        "//tensorflow/tsl/profiler/rpc:profiler_service_impl",
        "//tensorflow/tsl/profiler/utils:time_utils_impl",
    ] + tf_protos_profiler_service(),
)
