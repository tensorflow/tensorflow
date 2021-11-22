# Contains graph rewrites for TPU runtimes and optimizations.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)

package(
    default_visibility = [
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/stream_executor/tpu:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_rewrite_pass_registration",
    srcs = ["tpu_rewrite_pass_registration.cc"],
    deps = [
        ":distributed_tpu_configuration_rewrite_pass",
        ":distributed_tpu_rewrite_pass",
        ":encapsulate_tpu_computations_pass",
        ":variable_merger_pass",
        "//tensorflow/core:core_cpu",
    ],
    alwayslink = 1,
)

cc_library(
    name = "distributed_tpu_configuration_rewrite_pass",
    srcs = [
        "distributed_tpu_configuration_rewrite_pass.cc",
    ],
    hdrs = [
        "distributed_tpu_configuration_rewrite_pass.h",
    ],
    deps = [
        ":distributed_tpu_rewrite_helpers",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/tpu:tpu_init_mode",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_options",
    ],
)

cc_library(
    name = "distributed_tpu_rewrite_helpers",
    srcs = ["distributed_tpu_rewrite_helpers.cc"],
    hdrs = ["distributed_tpu_rewrite_helpers.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/tpu:tpu_defs",
    ],
)

cc_library(
    name = "variable_merger_pass",
    srcs = ["variable_merger_pass.cc"],
    hdrs = ["variable_merger_pass.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:optimization_registry",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "encapsulate_tpu_computations_pass",
    srcs = [
        "encapsulate_tpu_computations_pass.cc",
    ],
    hdrs = [
        "encapsulate_tpu_computations_pass.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/jit:compilation_passes",
        "//tensorflow/compiler/jit:encapsulate_util",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/tpu:tpu_compile_interface",
        "//tensorflow/core/tpu:tpu_defs",
    ] + if_static(
        [
            "//tensorflow/core/common_runtime:function",
            "//tensorflow/core/common_runtime:optimization_registry",
        ],
        [],
    ),
)

cc_library(
    name = "distributed_tpu_rewrite_pass_internal",
    srcs = ["distributed_tpu_rewrite_pass_internal.cc"],
    hdrs = ["distributed_tpu_rewrite_pass_internal.h"],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/random",
    ],
)

cc_library(
    name = "distributed_tpu_rewrite_pass",
    srcs = [
        "distributed_tpu_rewrite_pass.cc",
    ],
    hdrs = [
        "distributed_tpu_rewrite_pass.h",
    ],
    deps = [
        ":cond_builder",
        ":distributed_tpu_rewrite_helpers",
        ":distributed_tpu_rewrite_pass_internal",
        ":host_training_loop_optimization_util",
        ":incomplete_nodedef_builder",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/compiler/jit:encapsulate_util",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:resource_operation_table",
        "//tensorflow/compiler/tf2xla:sharding_util",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_cc",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/core/tpu:tpu_compile_interface",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_fingerprint_utils",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_platform_interface",
        "//tensorflow/stream_executor/tpu:tpu_topology_external",
    ] + if_static(
        [
            "//tensorflow/core/common_runtime:function",
            "//tensorflow/core/common_runtime:graph_constructor",
            "//tensorflow/core/common_runtime:lower_function_call_op",
            "//tensorflow/core/common_runtime:lower_functional_ops",
            "//tensorflow/core/common_runtime:lower_if_op",
            "//tensorflow/core/common_runtime:lower_while_op",
            "//tensorflow/core/common_runtime:optimization_registry",
        ],
        [
            "//tensorflow/core/common_runtime:core_cpu_base_no_ops",
        ],
    ),
)

cc_library(
    name = "incomplete_nodedef_builder",
    srcs = ["incomplete_nodedef_builder.cc"],
    hdrs = ["incomplete_nodedef_builder.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "cond_builder",
    srcs = ["cond_builder.cc"],
    hdrs = ["cond_builder.h"],
    deps = [
        ":incomplete_nodedef_builder",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "host_training_loop_optimization_util",
    srcs = [
        "host_training_loop_optimization_util.cc",
    ],
    hdrs = [
        "host_training_loop_optimization_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":distributed_tpu_rewrite_pass_internal",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "configure_tpu_embedding_rewrite_pass",
    srcs = [
        "configure_tpu_embedding_rewrite_pass.cc",
    ],
    hdrs = [
        "configure_tpu_embedding_rewrite_pass.h",
    ],
    deps = [
        ":distributed_tpu_rewrite_helpers",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_configuration_proto_rewrite",
    ],
)

cc_library(
    name = "configure_tpu_embedding_rewrite_registration",
    srcs = ["configure_tpu_embedding_rewrite_registration.cc"],
    deps = [
        ":configure_tpu_embedding_rewrite_pass",
        "//tensorflow/core:core_cpu",
    ],
    alwayslink = 1,
)
