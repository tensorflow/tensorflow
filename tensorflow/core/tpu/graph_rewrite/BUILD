# Contains graph rewrites for TPU runtimes and optimizations.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_rewrite_pass_registration",
    srcs = ["tpu_rewrite_pass_registration.cc"],
    deps = [
        ":combine_tpu_embedding_load_retrieve_pass",
        ":distributed_tpu_configuration_rewrite_pass",
        ":distributed_tpu_rewrite_pass",
        ":encapsulate_tpu_computations_pass",
        ":tpu_embedding_software_deduplication_rewrite_pass",
        ":update_tpu_embedding_ops_passes",
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
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/tpu:tpu_init_mode",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_options",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:logging",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "distributed_tpu_rewrite_helpers",
    srcs = ["distributed_tpu_rewrite_helpers.cc"],
    hdrs = ["distributed_tpu_rewrite_helpers.h"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:strcat",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "variable_merger_pass",
    srcs = ["variable_merger_pass.cc"],
    hdrs = ["variable_merger_pass.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:optimization_registry",
        "//tensorflow/core/framework:node_def_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
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
        "//tensorflow/compiler/jit:compilation_passes",
        "//tensorflow/compiler/jit:encapsulate_util",
        "//tensorflow/compiler/jit:xla_cluster_util",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:function_body",
        "//tensorflow/core/common_runtime:function_utils",
        "//tensorflow/core/config:flag_defs",
        "//tensorflow/core/tpu:tpu_compile_interface",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/tsl/util:env_var",
    ] + if_static(
        [
            "//tensorflow/core/common_runtime:function",
            "//tensorflow/core/common_runtime:optimization_registry",
        ],
        [],
    ),
)

tf_cc_test(
    name = "encapsulate_tpu_computations_pass_test",
    srcs = ["encapsulate_tpu_computations_pass_test.cc"],
    deps = [
        ":encapsulate_tpu_computations_pass",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:optimization_registry",
        "//tensorflow/core/config:flag_defs",
        "//tensorflow/core/framework:types_proto_cc",
    ],
)

cc_library(
    name = "distributed_tpu_rewrite_pass_internal",
    srcs = ["distributed_tpu_rewrite_pass_internal.cc"],
    hdrs = ["distributed_tpu_rewrite_pass_internal.h"],
    deps = ["@com_google_absl//absl/random"],
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
        "//tensorflow/compiler/jit:encapsulate_util",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/mlir/tensorflow:xla_sharding_util",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:resource_operation_table",
        "//tensorflow/compiler/tf2xla:sharding_util",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:device_propagation",
        "//tensorflow/core/platform:error_payloads",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_cc",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/core/tpu:tpu_compile_interface",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_fingerprint_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
        "@local_xla//xla:array2d",
        "@local_xla//xla:array3d",
        "@local_xla//xla:array4d",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/hlo/builder:sharding_builder",
        "@local_xla//xla/service:computation_placer",
        "@local_xla//xla/stream_executor/tpu:c_api_decl",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
        "@local_xla//xla/stream_executor/tpu:tpu_topology_external",
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
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "cond_builder",
    srcs = ["cond_builder.cc"],
    hdrs = ["cond_builder.h"],
    deps = [
        ":incomplete_nodedef_builder",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:status",
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
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:tstring",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "tpu_embedding_rewrite_pass_utils",
    srcs = ["tpu_embedding_rewrite_pass_utils.cc"],
    hdrs = ["tpu_embedding_rewrite_pass_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "combine_tpu_embedding_load_retrieve_pass",
    srcs = ["combine_tpu_embedding_load_retrieve_pass.cc"],
    hdrs = ["combine_tpu_embedding_load_retrieve_pass.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_embedding_rewrite_pass_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/core/tpu/ops:tpu_embedding_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:logging",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
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
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_configuration_proto_rewrite",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
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

cc_library(
    name = "tpu_embedding_software_deduplication_rewrite_pass",
    srcs = ["tpu_embedding_software_deduplication_rewrite_pass.cc"],
    hdrs = ["tpu_embedding_software_deduplication_rewrite_pass.h"],
    deps = [
        ":tpu_embedding_rewrite_pass_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_configuration_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "update_tpu_embedding_ops_passes",
    srcs = [
        "update_tpu_embedding_ops_passes.cc",
    ],
    hdrs = [
        "update_tpu_embedding_ops_passes.h",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//xla:status_macros",
    ],
)
