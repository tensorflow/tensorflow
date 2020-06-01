load("//third_party/mlir:tblgen.bzl", "gentbl")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_native_cc_binary")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//babelfish/device/...",
        "//learning/brain/experimental/dtensor/...",
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/google/xla/kernels/...",
        "//learning/brain/google/xla/mlir/...",
        "//learning/brain/swift/swift_mlir/...",
        "//learning/pathways/data_parallel/tf2xla/...",
        "//platforms/xla/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
    ],
)

exports_files(["ir/hlo_ops.td"])

filegroup(
    name = "hlo_ops_td_files",
    srcs = [
        "ir/chlo_ops.td",
        "ir/hlo_ops.td",
        "ir/hlo_ops_base.td",
        "ir/hlo_utils.td",
        "ir/lhlo_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)

gentbl(
    name = "chlo_ops_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "ir/chlo_ops.h.inc"),
        ("-gen-op-defs", "ir/chlo_ops.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/chlo_ops.td",
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "hlo_ops_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "ir/hlo_ops.h.inc"),
        ("-gen-op-defs", "ir/hlo_ops.cc.inc"),
        ("-gen-struct-attr-decls", "ir/hlo_structs.h.inc"),
        ("-gen-struct-attr-defs", "ir/hlo_structs.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/hlo_ops.td",
    td_includes = ["ir/hlo_utils.td"],
    td_srcs = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)

gentbl(
    name = "hlo_ops_base_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "ir/hlo_ops_base.h.inc"),
        ("-gen-op-defs", "ir/hlo_ops_base.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/hlo_ops_base.td",
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "lhlo_ops_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "ir/lhlo_ops.h.inc"),
        ("-gen-op-defs", "ir/lhlo_ops.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/lhlo_ops.td",
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "xla_legalize_tf_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_legalize_tf.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_tf_patterns.td",
    td_srcs = [
        ":hlo_ops_td_files",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "xla_canonicalize_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_canonicalize.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/canonicalize.td",
    td_srcs = [":hlo_ops_td_files"],
)

cc_library(
    name = "xla_legalize_tf",
    srcs = [
        "transforms/generated_legalize_tf.inc",
        "transforms/legalize_tf.cc",
        "transforms/legalize_tf_control_flow.cc",
    ],
    deps = [
        ":chlo_legalize_to_hlo",
        ":convert_op_folder",
        ":hlo",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:conv_grad_shape_utils",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_legalize_tf_with_tf2xla",
    srcs = ["transforms/legalize_tf_with_tf2xla.cc"],
    deps = [
        ":hlo",
        ":mlir_hlo_builder",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:export_tf_dialect_op",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/stream_executor:timer",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_sink_constants_to_control_flow",
    srcs = ["transforms/sink_constants_to_control_flow.cc"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "map_xla_to_scalar_op",
    hdrs = ["transforms/map_xla_to_scalar_op.h"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_hlo_to_lhlo_op",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "map_hlo_to_lhlo_op",
    hdrs = ["transforms/map_hlo_to_lhlo_op.h"],
    deps = [
        ":hlo",
        ":lhlo",
    ],
)

cc_library(
    name = "lhlo_legalize_to_affine",
    srcs = ["transforms/lhlo_legalize_to_affine.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_xla_to_scalar_op",
        "//tensorflow/compiler/xla:status",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_legalize_to_parallel_loops",
    srcs = ["transforms/lhlo_legalize_to_parallel_loops.cc"],
    deps = [
        ":lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_legalize_to_linalg",
    srcs = ["transforms/xla_legalize_to_linalg.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_xla_to_scalar_op",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_transform_unranked_hlo",
    srcs = ["transforms/xla_transform_unranked_hlo.cc"],
    deps = [
        ":hlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_legalize_to_gpu",
    srcs = ["transforms/lhlo_legalize_to_gpu.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_xla_to_scalar_op",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_fuse_linalg",
    srcs = ["transforms/lhlo_fuse_linalg.cc"],
    deps = [
        ":lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_copy_removal",
    srcs = ["transforms/lhlo_copy_removal.cc"],
    deps = [
        ":lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_legalize_to_lhlo",
    srcs = ["transforms/hlo_legalize_to_lhlo.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        ":map_hlo_to_lhlo_op",
        "@com_google_absl//absl/memory",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

gentbl(
    name = "xla_legalize_to_standard_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_legalize_to_standard.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_to_standard_patterns.td",
    td_srcs = [
        ":hlo_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    name = "xla_legalize_control_flow",
    srcs = ["transforms/legalize_control_flow.cc"],
    deps = [
        ":hlo",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_hlo_to_lhlo_with_xla",
    srcs = ["transforms/xla_hlo_to_lhlo_with_xla.cc"],
    hdrs = ["transforms/xla_hlo_to_lhlo_with_xla.h"],
    deps = [
        ":hlo",
        ":hlo_utils",
        ":lhlo",
        ":mlir_hlo_to_hlo",
        ":xla_dialect_registration",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_legalize_to_standard",
    srcs = ["transforms/legalize_to_standard.cc"],
    deps = [
        ":hlo",
        ":xla_legalize_to_standard_inc_gen",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)

gentbl(
    name = "xla_lower_complex_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_lower_complex.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/lower_complex_patterns.td",
    td_srcs = [
        ":hlo_ops_td_files",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    name = "xla_lower",
    srcs = [
        "transforms/generated_lower_complex.inc",
        "transforms/lower_complex.cc",
        "transforms/lower_general_dot.cc",
    ],
    deps = [
        ":hlo",
        ":xla_dialect_registration",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_materialize_broadcasts",
    srcs = ["transforms/materialize_broadcasts.cc"],
    deps = [
        ":hlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "xla_unfuse_batch_norm",
    srcs = ["transforms/unfuse_batch_norm.cc"],
    deps = [
        ":hlo",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "chlo_legalize_to_hlo",
    srcs = ["transforms/chlo_legalize_to_hlo.cc"],
    deps = [
        ":hlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "xla_test_passes",
    srcs = [
        "transforms/chlo_legalize_to_hlo_pass.cc",
        "transforms/materialize_broadcasts_pass.cc",
        "transforms/test_infer_shaped_type_pass.cc",
        "transforms/unfuse_batch_norm_pass.cc",
    ],
    deps = [
        ":chlo_legalize_to_hlo",  # build-cleaner: keep
        ":hlo",
        ":xla_materialize_broadcasts",  # build-cleaner: keep
        ":xla_unfuse_batch_norm",  # build-cleaner: keep
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo",
    srcs = [
        "ir/broadcast_utils.cc",
        "ir/chlo_ops.cc",
        "ir/hlo_ops.cc",
        "ir/hlo_ops.cc.inc",
        "ir/hlo_ops.h.inc",
        "ir/hlo_utils.cc",
    ],
    hdrs = [
        "ir/broadcast_utils.h",
        "ir/chlo_ops.h",
        "ir/hlo_ops.h",
        "ir/hlo_utils.h",
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    includes = ["include"],
    deps = [
        ":chlo_ops_inc_gen",
        ":convert_op_folder",
        ":hlo_ops_base_inc_gen",
        ":hlo_ops_inc_gen",
        ":xla_canonicalize_inc_gen",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_hlo_builder",
    srcs = ["ir/mlir_hlo_builder.cc"],
    hdrs = ["ir/mlir_hlo_builder.h"],
    deps = [
        ":attribute_importer",
        ":hlo",
        ":hlo_utils",
        ":type_to_shape",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "lhlo",
    srcs = [
        "ir/lhlo_ops.cc",
        "ir/lhlo_ops.cc.inc",
        "ir/lhlo_ops.h.inc",
    ],
    hdrs = [
        "ir/lhlo_ops.h",
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    includes = ["include"],
    deps = [
        ":hlo_ops_base_inc_gen",
        ":lhlo_ops_inc_gen",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.cc"],
    hdrs = ["hlo_utils.h"],
    includes = ["include"],
    deps = [
        ":convert_op_folder",
        ":hlo",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

# Library with XLA dialect static initialization.
cc_library(
    name = "xla_dialect_registration",
    srcs = ["ir/dialect_registration.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "type_to_shape",
    srcs = ["type_to_shape.cc"],
    hdrs = ["type_to_shape.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "type_to_shape_test",
    srcs = ["type_to_shape_test.cc"],
    deps = [
        ":type_to_shape",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mlir_hlo_to_hlo",
    srcs = [
        "mlir_hlo_to_hlo.cc",
        "operator_writers.inc",
    ],
    hdrs = ["mlir_hlo_to_hlo.h"],
    deps = [
        ":hlo",
        ":type_to_shape",
        ":xla_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/client/lib:quantize",
        "//tensorflow/compiler/xla/client/lib:slicing",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "hlo_to_mlir_hlo",
    srcs = ["hlo_to_mlir_hlo.cc"],
    hdrs = ["hlo_to_mlir_hlo.h"],
    deps = [
        ":hlo_module_importer",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_module_importer",
    srcs = [
        "hlo_function_importer.cc",
        "hlo_module_importer.cc",
    ],
    hdrs = [
        "hlo_function_importer.h",
        "hlo_module_importer.h",
    ],
    deps = [
        ":attribute_importer",
        ":hlo",
        ":hlo_utils",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_casting_utils",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "attribute_importer",
    srcs = ["attribute_importer.cc"],
    hdrs = ["attribute_importer.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core/platform:types",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "xla_mlir_translate",
    srcs = ["xla_mlir_translate.cc"],
    hdrs = ["xla_mlir_translate.h"],
    deps = [
        ":hlo_to_mlir_hlo",
        ":mlir_hlo_to_hlo",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

tf_native_cc_binary(
    name = "operator_writer_gen",
    srcs = ["operator_writer_gen.cc"],
    deps = [
        "@llvm-project//llvm:support",
        "@llvm-project//llvm:tablegen",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

genrule(
    name = "operator_writer_inc",
    srcs = [
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
        ":ir/hlo_ops.td",
        ":ir/hlo_ops_base.td",
        ":ir/hlo_utils.td",
    ],
    outs = ["operator_writers.inc"],
    cmd = ("$(location :operator_writer_gen) " +
           "-I external/llvm-project/mlir/include " +
           "-I external/org_tensorflow " +
           "$(location :ir/hlo_ops.td) " +
           " -o $@"),
    tools = [":operator_writer_gen"],
)

cc_library(
    name = "convert_op_folder",
    srcs = ["convert_op_folder.cc"],
    hdrs = ["convert_op_folder.h"],
    deps = [
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "all_xla_passes_for_testing",
    visibility = [
        "//tensorflow/compiler/mlir:__subpackages__",
    ],
    deps = [
        ":chlo_legalize_to_hlo",
        ":hlo",
        ":hlo_legalize_to_lhlo",
        ":lhlo",
        ":lhlo_copy_removal",
        ":lhlo_fuse_linalg",
        ":lhlo_legalize_to_affine",
        ":lhlo_legalize_to_gpu",
        ":lhlo_legalize_to_parallel_loops",
        ":xla_dialect_registration",
        ":xla_hlo_to_lhlo_with_xla",
        ":xla_legalize_control_flow",
        ":xla_legalize_tf",
        ":xla_legalize_tf_with_tf2xla",
        ":xla_legalize_to_linalg",
        ":xla_legalize_to_standard",
        ":xla_lower",
        ":xla_sink_constants_to_control_flow",
        ":xla_test_passes",
        ":xla_transform_unranked_hlo",
    ],
)

tf_cc_binary(
    name = "xla-opt",
    deps = [
        ":all_xla_passes_for_testing",
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/mlir:tf_mlir_opt_main",
    ],
)
