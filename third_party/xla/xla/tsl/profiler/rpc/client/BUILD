load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_protos_profiler_service",
    "tsl_cc_test",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla/tsl/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl/profiler:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "capture_profile",
    srcs = ["capture_profile.cc"],
    hdrs = ["capture_profile.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//xla/python:__pkg__",
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ]),
    deps = [
        ":profiler_client_for_pybind",
        ":remote_profiler_session_manager",
        ":save_profile",
        "//xla/tsl/profiler/convert:trace_events_to_json",
        "//xla/tsl/profiler/convert:xplane_to_trace_events",
        "//xla/tsl/profiler/utils:session_manager",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/protobuf:profiler_analysis_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "save_profile",
    srcs = ["save_profile.cc"],
    hdrs = ["save_profile.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//xla/python:__pkg__",
        "//xla/tsl/profiler:internal",
        "//xla/tsl/profiler/rpc:__pkg__",
        "//learning/pathways/data_parallel:__pkg__",
    ]),
    deps = [
        "//xla/tsl/lib/io:zlib_compression_options",
        "//xla/tsl/lib/io:zlib_outputbuffer",
        "//xla/tsl/profiler/utils:file_system_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
        "//xla:__subpackages__",
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ]),
    deps = [
        ":profiler_client_impl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
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
        "//xla/python:__pkg__",
        "//tensorflow/core/profiler/rpc/client:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ]),
    deps = [
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/protobuf:profiler_analysis_cc_grpc_proto",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
    ] + tsl_grpc_cc_dependencies(),
    alwayslink = True,
)

cc_library(
    name = "profiler_client_test_util",
    testonly = 1,
    hdrs = ["profiler_client_test_util.h"],
    deps = [
        "//xla/tsl/profiler/rpc:profiler_server_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:profiler_session",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
    ] + tf_protos_profiler_service(),
)

tsl_cc_test(
    name = "profiler_client_test",
    srcs = ["profiler_client_test.cc"],
    deps = [
        ":profiler_client",
        ":profiler_client_impl",  # for oss
        ":profiler_client_test_util",
        "//xla/tsl/profiler/rpc:profiler_server_impl",
        "//xla/tsl/profiler/rpc:profiler_service_impl",
        "//xla/tsl/profiler/utils:time_utils_impl",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:profiler_factory_impl",
        "@local_tsl//tsl/profiler/lib:profiler_session_impl",
    ] + tf_protos_profiler_service(),
)

cc_library(
    name = "remote_profiler_session_manager",
    srcs = ["remote_profiler_session_manager.cc"],
    hdrs = ["remote_profiler_session_manager.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":profiler_client_for_pybind",
        "//xla/tsl/profiler/utils:time_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:env_time",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/platform:types",
    ],
)

tsl_cc_test(
    name = "remote_profiler_session_manager_test",
    srcs = ["remote_profiler_session_manager_test.cc"],
    deps = [
        ":profiler_client_impl",  # for oss
        ":profiler_client_test_util",
        ":remote_profiler_session_manager",
        "//xla/tsl/profiler/rpc:profiler_server_impl",
        "//xla/tsl/profiler/rpc:profiler_service_impl",
        "//xla/tsl/profiler/utils:time_utils_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:profiler_factory_impl",
        "@local_tsl//tsl/profiler/lib:profiler_session_impl",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
    ] + tf_protos_profiler_service(),
)
