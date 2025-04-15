# copybara:uncomment(oss-unused) load("//net/grpc/go/build_defs:go_grpc_library.bzl", "go_grpc_library")
# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:build_config.bzl", "tf_additional_all_protos", "tf_proto_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//tensorflow:internal"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/profiler/...",
        "//tensorflow/python/eager/...",
        "//tensorflow/python/profiler/...",
        "//tensorflow/python/tpu/profiler/...",
    ],
)

tf_proto_library(
    name = "profiler_service_monitor_result_proto",
    srcs = ["profiler_service_monitor_result.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "@local_tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto",
    ],
    visibility = ["//visibility:public"],
    exports = [
        "@local_tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto",
    ],
)

tf_proto_library(
    name = "profiler_options_proto",
    srcs = ["profiler_options.proto"],
    make_default_target_header_only = True,
    protodeps = ["@local_tsl//tsl/profiler/protobuf:profiler_options_proto"],
    visibility = ["//visibility:public"],
    exports = ["@local_tsl//tsl/profiler/protobuf:profiler_options_proto"],
)

# This is needed because of how tf_android_core_proto_sources parses proto paths.
exports_files(
    srcs = ["profiler_options.proto"],
    visibility = ["//tensorflow/core:__pkg__"],
)

tf_proto_library(
    name = "profiler_service_proto",
    srcs = ["profiler_service.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "@local_tsl//tsl/profiler/protobuf:profiler_service_proto",
    ],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
    exports = ["@local_tsl//tsl/profiler/protobuf:profiler_service_proto"],
)

tf_proto_library(
    name = "profiler_analysis_proto",
    srcs = ["profiler_analysis.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "@local_tsl//tsl/profiler/protobuf:profiler_analysis_proto",
    ],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
    exports = ["@local_tsl//tsl/profiler/protobuf:profiler_analysis_proto"],
)

tf_proto_library(
    name = "protos_all",
    srcs = [
        "profile.proto",
        "tfprof_log.proto",
        "tfprof_options.proto",
        "tfprof_output.proto",
    ],
    protodeps = [
        "@local_tsl//tsl/profiler/protobuf:profile_proto",
    ] + tf_additional_all_protos(),
    visibility = [":friends"],
    exports = [
        "@local_tsl//tsl/profiler/protobuf:profile_proto",
    ],
)

cc_library(
    name = "profiler_impl",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/security/fuzzing/cc/ops:__pkg__",
    ],
    deps = [
        "//tensorflow/core/profiler/lib:profiler_factory_impl",
        "//tensorflow/core/profiler/lib:profiler_session_impl",
        "@local_xla//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
        "@local_xla//xla/tsl/profiler/backends/cpu:threadpool_listener",
        "@local_xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@local_xla//xla/tsl/profiler/utils:time_utils_impl",
    ],
    alwayslink = True,
)

# Libraries and binaries of TensorFlow profiler 1.x.
tf_cc_binary(
    name = "profiler",
    srcs = ["profiler.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":protos_all_cc",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/c:tf_status",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/profiler:tfprof_options",
        "//tensorflow/core/profiler/internal:tfprof_stats",
        "//tensorflow/core/profiler/internal:tfprof_utils",
        "//tensorflow/core/profiler/internal/advisor:tfprof_advisor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@linenoise",
    ],
)

cc_library(
    name = "tfprof_options",
    srcs = ["tfprof_options.cc"],
    hdrs = ["tfprof_options.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":protos_all_cc",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "//tensorflow/core/profiler/lib:mobile_srcs_no_runtime",
        "@local_tsl//tsl/profiler/lib:mobile_srcs_no_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "//tensorflow/core/profiler/lib:mobile_srcs_only_runtime",
        "@local_tsl//tsl/profiler/lib:mobile_srcs_only_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "profiler_analysis_proto_py_pb2",
#     has_services = 1,
#     visibility = ["//visibility:public"],
#     deps = [":profiler_analysis_proto"],
# )
#
# py_proto_library(
#     name = "protos_all_py_pb2",
#     visibility = [":friends"],
#     deps = [":protos_all"],
# )
# copybara:uncomment_end
