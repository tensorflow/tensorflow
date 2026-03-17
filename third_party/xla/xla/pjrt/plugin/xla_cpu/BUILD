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
    srcs = ["xla_cpu_pjrt_client.cc"],
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
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "cpu_memory",
    hdrs = ["cpu_memory.h"],
)

cc_library(
    name = "cpu_client_options",
    hdrs = ["cpu_client_options.h"],
    deps = [
        ":cpu_memory",
        ":cpu_topology_description",
        "//xla/backends/cpu/collectives:cpu_collectives",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/status:statusor",
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
        "//xla/backends/cpu:target_machine_options",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_device_dimensions",
        "//xla/runtime:device_id",
        "//xla/tsl/lib/strings:proto_serialization",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

tf_proto_library(
    name = "cpu_topology_proto",
    srcs = ["cpu_topology.proto"],
    protodeps = ["//xla/service/cpu:executable_proto"],
)

cc_library(
    name = "cpu_topology",
    srcs = ["cpu_topology.cc"],
    hdrs = ["cpu_topology.h"],
    deps = [
        ":cpu_topology_proto_cc",
        "//xla/backends/cpu:target_machine_options",
        "//xla/pjrt:pjrt_common",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "cpu_topology_description_test",
    srcs = ["cpu_topology_description_test.cc"],
    deps = [
        ":cpu_topology",
        ":cpu_topology_description",
        "//xla/backends/cpu:target_machine_options",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_dimensions",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "cpu_topology_test",
    srcs = ["cpu_topology_test.cc"],
    deps = [
        ":cpu_topology",
        ":cpu_topology_proto_cc",
        "//xla/backends/cpu:target_machine_options",
        "//xla/pjrt:pjrt_common",
        "//xla/runtime:device_id",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "cpu_static_registration",
    srcs = ["cpu_static_registration.cc"],
    deps = [
        "//xla/pjrt/c:pjrt_c_api_cpu_internal",
        "//xla/pjrt/plugin:plugin_names",
        "//xla/pjrt/plugin:static_registration",
    ],
    alwayslink = True,
)
