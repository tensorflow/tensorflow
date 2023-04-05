# copybara:uncomment(oss-unused) load("//net/grpc/go/build_defs:go_grpc_library.bzl", "go_grpc_library")
load("//tensorflow/tsl:tsl.bzl", "set_external_visibility")
load("//tensorflow/tsl/platform:build_config.bzl", "tf_proto_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

package_group(
    name = "friends",
    includes = [
        "//tensorflow/tsl/profiler:friends",
    ],
)

tf_proto_library(
    name = "xplane_proto",
    srcs = ["xplane.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    visibility = set_external_visibility([":friends"]),
)

tf_proto_library(
    name = "profiler_options_proto",
    srcs = ["profiler_options.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "profiler_service_monitor_result_proto",
    srcs = ["profiler_service_monitor_result.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "profiler_service_proto",
    srcs = ["profiler_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_grpc_library = True,
    create_java_proto = False,
    make_default_target_header_only = True,
    protodeps = [
        ":profiler_options_proto",
        ":profiler_service_monitor_result_proto",
    ],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "profiler_analysis_proto",
    srcs = ["profiler_analysis.proto"],
    has_services = 1,
    cc_api_version = 2,
    create_grpc_library = True,
    create_java_proto = False,
    make_default_target_header_only = True,
    protodeps = [":profiler_service_proto"],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "trace_events_proto",
    srcs = ["trace_events.proto"],
    cc_api_version = 2,
    visibility = set_external_visibility([":friends"]),
)

# This is needed because of how tf_android_core_proto_sources parses proto paths.
exports_files(
    srcs = ["xplane.proto"],
    visibility = set_external_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/tsl:__pkg__",
    ]),
)

tf_proto_library(
    name = "profile_proto",
    srcs = ["profile.proto"],
    cc_api_version = 2,
    visibility = set_external_visibility([
        "//tensorflow/compiler/xla/python:__pkg__",
        "//tensorflow/tsl/profiler:internal",
    ]),
)

tf_proto_library(
    name = "protos_all",
    create_go_proto = False,
    make_default_target_header_only = True,
    protodeps = [
        ":xplane_proto",
        ":profiler_options_proto",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "xplane_py_pb2",
#     api_version = 2,
#     visibility = set_external_visibility([":friends"]),
#     deps = [":xplane_proto"],
# )
# copybara:uncomment_end
