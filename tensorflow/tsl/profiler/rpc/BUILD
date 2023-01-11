load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/tsl/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)
load("//tensorflow/tsl:tsl.default.bzl", "tsl_grpc_cc_dependencies")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/tsl/profiler:internal"],
    licenses = ["notice"],
)

# Linked to pywrap_tensorflow.
cc_library(
    name = "profiler_service_impl",
    srcs = ["profiler_service_impl.cc"],
    hdrs = ["profiler_service_impl.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/data/service:__pkg__",
        "//tensorflow/core/distributed_runtime/rpc:__pkg__",
        "//tensorflow/core/profiler/rpc:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/tsl/profiler/rpc/client:__pkg__",
        "//tensorflow_serving/model_servers:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_time",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/lib:profiler_session",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/rpc/client:save_profile",
        "//tensorflow/tsl/profiler/utils:file_system_utils",
        "//tensorflow/tsl/profiler/utils:math_utils",
        "//tensorflow/tsl/profiler/utils:time_utils",
        "//tensorflow/tsl/profiler/utils:xplane_utils",
    ] + tsl_grpc_cc_dependencies(),
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_server_for_pybind",
    actual = ":profiler_server_impl",
    visibility = ["//tensorflow/python/profiler/internal:__pkg__"],
)

cc_library(
    name = "profiler_server_impl",
    srcs = ["profiler_server.cc"],
    hdrs = ["profiler_server.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/compiler/xla:__subpackages__",
        "//tensorflow/core/profiler/rpc:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
        "//tensorflow/tsl/profiler:internal",
        "//tensorflow/tsl/profiler/rpc/client:__pkg__",
    ],
    deps = [
        ":profiler_service_impl",
        "@com_google_absl//absl/strings",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
    ] + tsl_grpc_cc_dependencies(),
    alwayslink = True,
)
