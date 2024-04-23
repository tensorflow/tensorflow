load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")
load(
    "//tsl/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//tsl/profiler:internal"]),
    licenses = ["notice"],
)

# Linked to pywrap_tensorflow.
cc_library(
    name = "profiler_service_impl",
    srcs = ["profiler_service_impl.cc"],
    hdrs = ["profiler_service_impl.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tensorflow/core/data/service:__pkg__",
        "//tensorflow/core/distributed_runtime/rpc:__pkg__",
        "//tensorflow/core/profiler/rpc:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tsl/profiler/rpc/client:__pkg__",
        "//tensorflow_serving/model_servers:__pkg__",
    ]),
    deps = [
        "//tsl/platform:env",
        "//tsl/platform:env_time",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/profiler/lib:profiler_session",
        "//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "//tsl/profiler/protobuf:profiler_service_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "//tsl/profiler/rpc/client:save_profile",
        "//tsl/profiler/utils:file_system_utils",
        "//tsl/profiler/utils:math_utils",
        "//tsl/profiler/utils:time_utils",
        "//tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ] + tsl_grpc_cc_dependencies(),
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_server_for_pybind",
    actual = ":profiler_server_impl",
    visibility = internal_visibility(["//tensorflow/python/profiler/internal:__pkg__"]),
)

cc_library(
    name = "profiler_server_impl",
    srcs = ["profiler_server.cc"],
    hdrs = ["profiler_server.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@local_xla//xla:__subpackages__",
        "//tensorflow/core/profiler/rpc:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
        "//tsl/profiler:internal",
        "//tsl/profiler/rpc/client:__pkg__",
    ]),
    deps = [
        ":profiler_service_impl",
        "//tsl/platform:logging",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "@com_google_absl//absl/strings",
    ] + tsl_grpc_cc_dependencies(),
    alwayslink = True,
)
