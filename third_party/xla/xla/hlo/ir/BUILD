# Description:
#   XLA’s HLO Intermediate Representation implementation.

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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

cc_library(
    name = "hlo",
    srcs = [
        "collective_device_list.cc",
        "dfs_hlo_visitor.cc",
        "dynamic_parameter_binding.cc",
        "hlo_computation.cc",
        "hlo_input_output_alias_config.cc",
        "hlo_instruction.cc",
        "hlo_instructions.cc",
        "hlo_module.cc",
        "hlo_module_metadata.cc",
        "hlo_opcode.cc",
        "hlo_original_value.cc",
        "hlo_schedule.cc",
        "hlo_sharding_metadata.cc",
    ],
    hdrs = [
        "collective_device_list.h",
        "dfs_hlo_visitor.h",
        "dfs_hlo_visitor_with_default.h",
        "dynamic_parameter_binding.h",
        "hlo_casting_utils.h",
        "hlo_clone_context.h",
        "hlo_computation.h",
        "hlo_domain_metadata.h",
        "hlo_input_output_alias_config.h",
        "hlo_instruction.h",
        "hlo_instructions.h",
        "hlo_module.h",
        "hlo_module_metadata.h",
        "hlo_op_metadata.h",
        "hlo_opcode.h",
        "hlo_original_value.h",
        "hlo_print_options.h",
        "hlo_schedule.h",
        "hlo_sharding.h",
        "hlo_sharding_metadata.h",
    ],
    deps = [
        ":backend_config",
        ":hlo_op_metadata",
        ":hlo_sharding",
        ":ptrvec",
        ":tile_assignment",
        "//xla:array",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_pool",
        "//xla:literal_util",
        "//xla:online_topsort",
        "//xla:printer",
        "//xla:protobuf_util",
        "//xla:shape_layout",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:sort_json",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/parser:hlo_lexer",
        "//xla/service:compilation_environments",
        "//xla/service:computation_layout",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:mapped_ptr_container_sorter",
        "//xla/service:metrics_proto_cc",
        "//xla/service:name_uniquer",
        "//xla/tsl/lib/gtl:iterator_range",
        "//xla/tsl/lib/gtl:map_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@highwayhash",
        "@highwayhash//:arch_specific",
        "@highwayhash//:hh_types",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "hlo_opcode_test",
    srcs = ["hlo_opcode_test.cc"],
    deps = [
        ":hlo",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "hlo_computation_test",
    srcs = ["hlo_computation_test.cc"],
    deps = [
        ":hlo",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "hlo_instruction_test",
    srcs = ["hlo_instruction_test.cc"],
    deps = [
        ":hlo",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:xla_data_proto_cc",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "hlo_sharding",
    srcs = ["hlo_sharding.cc"],
    hdrs = ["hlo_sharding.h"],
    deps = [
        ":hlo_op_metadata",
        ":tile_assignment",
        "//xla:array",
        "//xla:printer",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "hlo_op_metadata",
    srcs = ["hlo_op_metadata.cc"],
    hdrs = ["hlo_op_metadata.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "hlo_module_test",
    srcs = ["hlo_module_test.cc"],
    deps = [
        ":hlo",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_module_config",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "backend_config",
    srcs = ["backend_config.cc"],
    hdrs = ["backend_config.h"],
    deps = [
        "//xla:util",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:human_readable_json",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "backend_config_test",
    srcs = ["backend_config_test.cc"],
    deps = [
        ":backend_config",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "hlo_module_group",
    srcs = ["hlo_module_group.cc"],
    hdrs = ["hlo_module_group.h"],
    deps = [
        ":hlo",
        "//xla:status_macros",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_instruction_utils",
    srcs = ["hlo_instruction_utils.cc"],
    hdrs = ["hlo_instruction_utils.h"],
    deps = [
        ":hlo",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "hlo_instruction_utils_test",
    srcs = ["hlo_instruction_utils_test.cc"],
    deps = [
        ":hlo",
        ":hlo_instruction_utils",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_query",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "ptrvec",
    hdrs = ["ptrvec.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/log:check",
    ],
)

xla_cc_test(
    name = "ptrvec_test",
    srcs = ["ptrvec_test.cc"],
    deps = [
        ":ptrvec",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tile_assignment",
    srcs = ["tile_assignment.cc"],
    hdrs = ["tile_assignment.h"],
    deps = [
        "//xla:array",
        "//xla:printer",
        "//xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "collective_device_list_test",
    srcs = ["collective_device_list_test.cc"],
    deps = [
        ":hlo",
        "//xla:xla_data_proto_cc",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "hlo_casting_utils_test",
    srcs = ["hlo_casting_utils_test.cc"],
    deps = [
        ":hlo",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:test_main",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest",
    ],
)
