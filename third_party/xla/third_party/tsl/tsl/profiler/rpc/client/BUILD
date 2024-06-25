load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load(
    "//tsl/platform:build_config.bzl",
    "tf_protos_profiler_service",
    "tsl_cc_test",
)
load(
    "//tsl/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tsl/profiler:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "capture_profile",
    srcs = ["capture_profile.cc"],
    hdrs = ["capture_profile.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@local_xla//xla/python:__pkg__",
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
        "//learning/pathways/data_parallel:__pkg__",
    ]),
    deps = [
        ":profiler_client_for_pybind",
        ":remote_profiler_session_manager",
        ":save_profile",
        "//tsl/platform:errors",
        "//tsl/platform:platform_port",
        "//tsl/platform:status",
        "//tsl/platform:types",
        "//tsl/profiler/convert:trace_events_to_json",
        "//tsl/profiler/convert:xplane_to_trace_events",
        "//tsl/profiler/protobuf:profiler_analysis_proto_cc",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tsl/profiler/protobuf:profiler_service_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "//tsl/profiler/utils:session_manager",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "save_profile",
    srcs = ["save_profile.cc"],
    hdrs = ["save_profile.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "@local_xla//xla/python:__pkg__",
        "//tsl/profiler:internal",
        "//tsl/profiler/rpc:__pkg__",
    ]),
    deps = [
        "//tsl/lib/io:zlib_compression_options",
        "//tsl/lib/io:zlib_outputbuffer",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:status",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:profiler_service_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "//tsl/profiler/utils:file_system_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_client_for_pybind",
    actual = ":profiler_client",
    visibility = internal_visibility([
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ]),
)

cc_library(
    name = "profiler_client",
    hdrs = ["profiler_client.h"],
    visibility = internal_visibility([
        "@local_xla//xla:__subpackages__",
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ]),
    deps = [
        ":profiler_client_impl",
        "//tsl/platform:status",
        "//tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
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
    visibility = internal_visibility([
        "@local_xla//xla/python:__pkg__",
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ]),
    deps = [
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:status",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "//tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ] + tsl_grpc_cc_dependencies(),
    alwayslink = True,
)

cc_library(
    name = "profiler_client_test_util",
    testonly = 1,
    hdrs = ["profiler_client_test_util.h"],
    deps = [
        "//tsl/platform:logging",
        "//tsl/platform:test",
        "//tsl/platform:types",
        "//tsl/profiler/lib:profiler_session",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tsl/profiler/rpc:profiler_server_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ] + tf_protos_profiler_service(),
)

tsl_cc_test(
    name = "profiler_client_test",
    srcs = ["profiler_client_test.cc"],
    deps = [
        ":profiler_client",
        ":profiler_client_impl",  # for oss
        ":profiler_client_test_util",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
        "//tsl/profiler/lib:profiler_factory_impl",
        "//tsl/profiler/lib:profiler_session_impl",
        "//tsl/profiler/rpc:profiler_server_impl",
        "//tsl/profiler/rpc:profiler_service_impl",
        "//tsl/profiler/utils:time_utils_impl",
        "@com_google_absl//absl/time",
    ] + tf_protos_profiler_service(),
)

cc_library(
    name = "remote_profiler_session_manager",
    srcs = ["remote_profiler_session_manager.cc"],
    hdrs = ["remote_profiler_session_manager.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":profiler_client_for_pybind",
        "//tsl/platform:env_time",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "//tsl/profiler/utils:time_utils",
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
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
        "//tsl/profiler/lib:profiler_factory_impl",
        "//tsl/profiler/lib:profiler_session_impl",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tsl/profiler/rpc:profiler_server_impl",
        "//tsl/profiler/rpc:profiler_service_impl",
        "//tsl/profiler/utils:time_utils_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ] + tf_protos_profiler_service(),
)
