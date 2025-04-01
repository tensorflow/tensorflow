load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_cpu_pjrt_client",
    srcs = [
        "xla_cpu_pjrt_client.cc",
    ],
    hdrs = ["xla_cpu_pjrt_client.h"],
    deps = [
        ":cpu_client_options",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/cpu:cpu_client",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "xla_cpu_pjrt_client_test",
    srcs = ["xla_cpu_pjrt_client_test.cc"],
    deps = [
        ":xla_cpu_pjrt_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "cpu_client_options",
    srcs = [],
    hdrs = ["cpu_client_options.h"],
    deps = [
        "//xla/backends/cpu/collectives:cpu_collectives",
        "//xla/service:hlo_module_config",
    ],
)

cc_library(
    name = "cpu_device_description",
    srcs = ["cpu_device_description.cc"],
    hdrs = ["cpu_device_description.h"],
    deps = [
        ":cpu_topology",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "cpu_execute_options",
    srcs = [],
    hdrs = ["cpu_execute_options.h"],
    deps = [
        "//xla/backends/cpu/collectives:cpu_collectives",
        "//xla/pjrt:pjrt_executable",
    ],
)

cc_library(
    name = "cpu_topology_description",
    srcs = ["cpu_topology_description.cc"],
    hdrs = ["cpu_topology_description.h"],
    deps = [
        ":cpu_device_description",
        ":cpu_topology",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/tsl/lib/strings:proto_serialization",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

tf_proto_library(
    name = "cpu_topology_proto",
    srcs = ["cpu_topology.proto"],
)

cc_library(
    name = "cpu_topology",
    srcs = ["cpu_topology.cc"],
    hdrs = ["cpu_topology.h"],
    deps = [
        ":cpu_topology_proto_cc",
        "//xla/pjrt:pjrt_common",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "cpu_topology_test",
    srcs = ["cpu_topology_test.cc"],
    deps = [
        ":cpu_topology",
        ":cpu_topology_proto_cc",
        "//xla/pjrt:pjrt_common",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
    ],
)
