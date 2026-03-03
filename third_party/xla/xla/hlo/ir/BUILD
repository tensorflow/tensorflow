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
        "hlo_original_value_util.cc",
        "hlo_schedule.cc",
        "hlo_sharding_metadata.cc",
        "replica_group.cc",
        "stack_frames.cc",
    ],
    hdrs = [
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
        "hlo_original_value_util.h",
        "hlo_print_options.h",
        "hlo_schedule.h",
        "hlo_sharding.h",
        "hlo_sharding_metadata.h",
        "replica_group.h",
        "stack_frames.h",
    ],
    deps = [
        ":backend_config",
        ":hlo_sharding",
        ":mesh_and_axis",
        ":named_sharding",
        ":ptrvec",
        ":tile_assignment",
        "//xla:array",
        "//xla:comparison_util",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:literal_pool",
        "//xla:literal_util",
        "//xla:online_topsort",
        "//xla:printer",
        "//xla:protobuf_util",
        "//xla:shape_layout",
        "//xla:shape_pool",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:sort_json",
        "//xla:status_macros",
        "//xla:tuple_tree",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/parser:hlo_lexer",
        "//xla/hlo/utils:pointer_utils",
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
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/functional:overload",
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
        "@com_google_protobuf//:protobuf_lite",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:protobuf",
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
    name = "stack_frames_test",
    srcs = ["stack_frames_test.cc"],
    deps = [
        ":hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
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
        "//xla:printer",
        "//xla:shape_util",
        "//xla:side_effect_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mesh_and_axis",
    srcs = ["mesh_and_axis.cc"],
    hdrs = ["mesh_and_axis.h"],
    deps = [
        ":tile_assignment",
        "//xla:array",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "named_sharding",
    srcs = ["named_sharding.cc"],
    hdrs = ["named_sharding.h"],
    deps = [
        ":hlo_op_metadata",
        ":mesh_and_axis",
        ":tile_assignment",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_sharding",
    srcs = ["hlo_sharding.cc"],
    hdrs = ["hlo_sharding.h"],
    deps = [
        ":hlo_op_metadata",
        ":mesh_and_axis",
        ":named_sharding",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:buffer_assignment",
        "//xla/service:buffer_value",
        "//xla/service:compilation_environments",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_util",
        "//xla/service:logical_buffer",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:human_readable_json",
        "@tsl//tsl/platform:protobuf",
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
    name = "hlo_instruction_utils",
    srcs = ["hlo_instruction_utils.cc"],
    hdrs = ["hlo_instruction_utils.h"],
    deps = [
        ":hlo",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "tile_assignment_test",
    srcs = ["tile_assignment_test.cc"],
    deps = [
        ":tile_assignment",
        "//xla:array",
        "//xla:array2d",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "replica_group_test",
    srcs = ["replica_group_test.cc"],
    deps = [
        ":hlo",
        ":mesh_and_axis",
        ":tile_assignment",
        "//xla:array",
        "//xla:array2d",
        "//xla:xla_data_proto_cc",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "mesh_and_axis_test",
    srcs = ["mesh_and_axis_test.cc"],
    deps = [
        ":mesh_and_axis",
        ":tile_assignment",
        "//xla:array",
        "//xla:array2d",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "named_sharding_test",
    srcs = ["named_sharding_test.cc"],
    deps = [
        ":mesh_and_axis",
        ":named_sharding",
        ":tile_assignment",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_googletest//:gtest_main",
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

cc_library(
    name = "collective_op_group_mode",
    srcs = ["collective_op_group_mode.cc"],
    hdrs = ["collective_op_group_mode.h"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "collective_op_group_mode_test",
    srcs = ["collective_op_group_mode_test.cc"],
    deps = [
        ":collective_op_group_mode",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "dfs_hlo_visitor_test",
    srcs = ["dfs_hlo_visitor_test.cc"],
    deps = [
        ":hlo",
        "//xla:literal_util",
        "//xla:status_macros",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_module_config",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "hlo_original_value_test",
    srcs = ["hlo_original_value_test.cc"],
    deps = [
        ":hlo",
        "//xla:shape_util",
        "//xla:tuple_tree",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_googletest//:gtest_main",
    ],
)
