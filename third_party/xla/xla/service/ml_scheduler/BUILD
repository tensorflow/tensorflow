# Description:
#   HLO Schedule data collection, demonstration records, and modeling support for XLA.

load(
    "//xla:xla.default.bzl",
    "xla_py_proto_library",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

tf_proto_library(
    name = "hardware_topology_proto",
    srcs = ["hardware_topology.proto"],
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

xla_py_proto_library(
    name = "hardware_topology_proto_py_pb2",
    deps = [":hardware_topology_proto"],
)

tf_proto_library(
    name = "pjrt_executable_bundle_proto",
    srcs = ["pjrt_executable_bundle.proto"],
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
    deps = [
        ":hardware_topology_proto",
        "//xla:xla_data_proto",
        "//xla:xla_proto",
    ],
)

xla_py_proto_library(
    name = "pjrt_executable_bundle_proto_py_pb2",
    deps = [":pjrt_executable_bundle_proto"],
)

tf_proto_library(
    name = "hlo_benchmark_proto",
    srcs = ["hlo_benchmark.proto"],
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_executable_bundle_proto",
        "//xla:xla_data_proto",
        "//xla/service:hlo_proto",
    ],
)

xla_py_proto_library(
    name = "hlo_benchmark_proto_py_pb2",
    deps = [":hlo_benchmark_proto"],
)

tf_proto_library(
    name = "hlo_schedule_record_proto",
    srcs = ["hlo_schedule_record.proto"],
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
    deps = [
        ":hardware_topology_proto",
        ":hlo_benchmark_proto",
        ":pjrt_executable_bundle_proto",
        "//xla:xla_proto",
        "//xla/hlo/transforms/simplifiers:memory_scheduler_proto",
        "//xla/service:hlo_proto",
    ],
)

xla_py_proto_library(
    name = "hlo_schedule_record_proto_py_pb2",
    deps = [":hlo_schedule_record_proto"],
)
