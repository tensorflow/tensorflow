load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "import_model",
    srcs = [
        "import_model.cc",
    ],
    hdrs = [
        "export_graphdef.h",
        "import_model.h",
    ],
    deps = [
        ":mlir_roundtrip_flags",
        ":upgrade_graph",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:loader_lite",
        "//tensorflow/cc/saved_model:loader_util",
        "//tensorflow/compiler/jit:shape_inference_helpers",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/mlir/tensorflow:convert_attr",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/mlir/tensorflow:xla_sharding_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:initialize_variables_in_session_init",
        "//tensorflow/compiler/mlir/tensorflow/transforms:lift_variables_lib",
        "//tensorflow/compiler/mlir/tensorflow/transforms:mark_initialized_variables_lib",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_saved_model_passes",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "//tensorflow/core/platform:crash_analysis",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/client:sharding_builder",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_parser",
    ],
)

tf_cc_test(
    name = "tf_mlir_translate_registration_test",
    size = "small",
    srcs = ["tf_mlir_translate_registration_test.cc"],
    deps = [
        ":translate_registration",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "export_tf_dialect_op",
    srcs = [
        "export_tf_dialect_op.cc",
    ],
    hdrs = [
        "export_tf_dialect_op.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:export_utils",
        "//tensorflow/compiler/mlir/utils:string_container_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:IR",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "translate_tf_dialect_op",
    srcs = ["translate_tf_dialect_op.cc"],
    deps = [
        ":export_tf_dialect_op",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:protobuf",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_pass",
    srcs = ["mlir_roundtrip_pass.cc"],
    hdrs = ["mlir_roundtrip_pass.h"],
    deps = [
        ":export_graphdef",
        ":import_model",
        ":mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "mlir_roundtrip_pass_registration",
    srcs = ["mlir_roundtrip_pass_registration.cc"],
    deps = [
        ":mlir_roundtrip_pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_flags",
    srcs = ["mlir_roundtrip_flags.cc"],
    hdrs = ["mlir_roundtrip_flags.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "mlir_import_options",
    hdrs = ["mlir_import_options.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "translate_lib",
    srcs = ["tf_mlir_translate.cc"],
    hdrs = ["tf_mlir_translate.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":import_model",
        ":mlir_roundtrip_flags",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:loader_lite",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "//tensorflow/core/util/tensor_bundle:byteswaptensor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "translate_cl_options",
    srcs = [
        "tf_mlir_translate_cl.cc",
    ],
    hdrs = [
        "tf_mlir_translate_cl.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "export_graphdef",
    srcs = [
        "export_graphdef.cc",
    ],
    hdrs = [
        "export_graphdef.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":export_tf_dialect_op",
        ":mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:export_utils",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/mlir/tensorflow:verify_suitable_for_graph_export",
        "//tensorflow/compiler/mlir/utils:name_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/graph/regularization:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "translate_registration",
    srcs = [
        "tf_mlir_translate_registration.cc",
    ],
    deps = [
        ":export_graphdef",
        ":mlir_roundtrip_flags",
        ":translate_cl_options",
        ":translate_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/service/cpu:cpu_compiler",
        "@local_xla//xla/service/cpu:cpu_transfer_manager",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@local_xla//xla/stream_executor/host:host_platform_id",
    ],
    alwayslink = 1,
)

cc_library(
    name = "split_into_island_per_op_pass",
    srcs = ["split_into_island_per_op_pass.cc"],
    hdrs = [
        "split_into_island_per_op_pass.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_executor_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_pass_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "upgrade_graph",
    srcs = ["upgrade_graph.cc"],
    hdrs = ["upgrade_graph.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@llvm-project//llvm:Support",
    ],
)
