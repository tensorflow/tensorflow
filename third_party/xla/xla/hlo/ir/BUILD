# Description:
#   XLA’s HLO Intermediate Representation implementation.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
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
        "hlo_frontend_attributes.cc",
        "hlo_input_output_alias_config.cc",
        "hlo_instruction.cc",
        "hlo_instructions.cc",
        "hlo_module.cc",
        "hlo_module_metadata.cc",
        "hlo_op_metadata.cc",
        "hlo_opcode.cc",
        "hlo_schedule.cc",
        "hlo_sharding.cc",
        "hlo_sharding_metadata.cc",
    ],
    hdrs = [
        "dfs_hlo_visitor.h",
        "dfs_hlo_visitor_with_default.h",
        "dynamic_parameter_binding.h",
        "hlo_casting_utils.h",
        "hlo_clone_context.h",
        "hlo_computation.h",
        "hlo_domain_metadata.h",
        "hlo_frontend_attributes.h",
        "hlo_input_output_alias_config.h",
        "hlo_instruction.h",
        "hlo_instructions.h",
        "hlo_module.h",
        "hlo_module_metadata.h",
        "hlo_op_metadata.h",
        "hlo_opcode.h",
        "hlo_schedule.h",
        "hlo_sharding.h",
        "hlo_sharding_metadata.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":tile_assignment",
        "//xla:array",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:printer",
        "//xla:protobuf_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/service:compilation_environments",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_lexer",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:mapped_ptr_container_sorter",
        "//xla/service:name_uniquer",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/gtl:iterator_range",
        "@local_tsl//tsl/lib/gtl:map_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:human_readable_json",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_module_group",
    srcs = ["hlo_module_group.cc"],
    hdrs = ["hlo_module_group.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_reachability",
    srcs = ["hlo_reachability.cc"],
    hdrs = ["hlo_reachability.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":hlo",
        "//xla:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tile_assignment",
    srcs = ["tile_assignment.cc"],
    hdrs = ["tile_assignment.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:array",
        "//xla:printer",
        "//xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
    ],
)
