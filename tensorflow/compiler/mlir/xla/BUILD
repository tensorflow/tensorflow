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
        "//learning/pathways/data_parallel/tf2xla/...",
        "//platforms/xla/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
    ],
)

gentbl(
    name = "xla_legalize_tf_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_legalize_tf.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_tf_patterns.td",
    td_relative_includes = [
        "../hlo/include",
    ],
    td_srcs = [
        "//tensorflow/compiler/mlir/hlo:hlo_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

cc_library(
    name = "xla_legalize_tf",
    srcs = [
        "transforms/generated_legalize_tf.inc",
        "transforms/legalize_tf.cc",
        "transforms/legalize_tf_control_flow.cc",
    ],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":chlo_legalize_to_hlo",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:convert_op_folder",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/client/lib:conv_grad_size_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:conv_grad_shape_utils",
        "//tensorflow/core/lib/bfloat16",
        "@llvm-project//llvm:Support",
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
        ":mlir_hlo_builder",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/hlo",
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
        "@llvm-project//llvm:Support",
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
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "@llvm-project//llvm:Support",
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
        ":map_hlo_to_lhlo_op",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "map_hlo_to_lhlo_op",
    hdrs = ["transforms/map_hlo_to_lhlo_op.h"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
    ],
)

cc_library(
    name = "lhlo_legalize_to_affine",
    srcs = ["transforms/lhlo_legalize_to_affine.cc"],
    deps = [
        ":map_xla_to_scalar_op",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/xla:status",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
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
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
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
    name = "lhlo_legalize_to_llvm",
    srcs = ["transforms/lhlo_legalize_to_llvm.cc"],
    deps = [
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_legalize_to_linalg",
    srcs = ["transforms/xla_legalize_to_linalg.cc"],
    hdrs = ["transforms/rewriters.h"],
    deps = [
        ":map_xla_to_scalar_op",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
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
    hdrs = ["transforms/rewriters.h"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
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
        ":map_xla_to_scalar_op",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
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
    hdrs = ["transforms/passes.h"],
    deps = [
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_copy_removal",
    srcs = ["transforms/lhlo_copy_removal.cc"],
    hdrs = ["transforms/passes.h"],
    deps = [
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_legalize_to_lhlo",
    srcs = ["transforms/hlo_legalize_to_lhlo.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        ":map_hlo_to_lhlo_op",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cycle_detector",
    srcs = ["transforms/cycle_detector.cc"],
    hdrs = ["transforms/cycle_detector.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "cycle_detector_test",
    srcs = ["transforms/cycle_detector_test.cc"],
    deps = [
        ":cycle_detector",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "xla_hlo_fusion",
    srcs = ["transforms/xla_hlo_fusion.cc"],
    deps = [
        ":cycle_detector",
        "//tensorflow/compiler/mlir/hlo",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
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
    td_relative_includes = [
        "../hlo/include",
    ],
    td_srcs = [
        "//tensorflow/compiler/mlir/hlo:hlo_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_library(
    name = "xla_legalize_control_flow",
    srcs = ["transforms/legalize_control_flow.cc"],
    hdrs = ["transforms/passes.h"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "@llvm-project//llvm:Support",
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
        ":hlo_module_importer",
        ":hlo_utils",
        ":mlir_hlo_to_hlo",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_casting_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_legalize_to_standard",
    srcs = ["transforms/legalize_to_standard.cc"],
    hdrs = ["transforms/passes.h"],
    deps = [
        ":xla_legalize_tanh_to_approximation",
        ":xla_legalize_to_standard_inc_gen",
        "//tensorflow/compiler/mlir/hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_legalize_tanh_to_approximation",
    srcs = ["transforms/legalize_tanh_to_approximation.cc"],
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
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
    td_relative_includes = [
        "../hlo/include",
    ],
    td_srcs = [
        "//tensorflow/compiler/mlir/hlo:hlo_ops_td_files",
        "@llvm-project//llvm:Support",
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
    hdrs = [
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
        "@llvm-project//llvm:Support",
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
        "//tensorflow/compiler/mlir/hlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "xla_unfuse_batch_norm",
    srcs = ["transforms/unfuse_batch_norm.cc"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "chlo_legalize_to_hlo",
    srcs = ["transforms/chlo_legalize_to_hlo.cc"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "xla_test_passes",
    srcs = [
        "transforms/chlo_legalize_to_hlo_pass.cc",
        "transforms/lhlo_legalize_to_llvm_pass.cc",
        "transforms/materialize_broadcasts_pass.cc",
        "transforms/rewriters.h",
        "transforms/test_infer_shaped_type_pass.cc",
        "transforms/unfuse_batch_norm_pass.cc",
    ],
    deps = [
        ":chlo_legalize_to_hlo",  # build-cleaner: keep
        ":lhlo_legalize_to_llvm",  # build-cleaner: keep
        ":xla_materialize_broadcasts",  # build-cleaner: keep
        ":xla_unfuse_batch_norm",  # build-cleaner: keep
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
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
        ":hlo_module_importer",
        ":hlo_utils",
        ":type_to_shape",
        "//tensorflow/compiler/mlir/hlo",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.cc"],
    hdrs = ["hlo_utils.h"],
    includes = ["include"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:convert_op_folder",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "type_to_shape",
    srcs = ["type_to_shape.cc"],
    hdrs = ["type_to_shape.h"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "@llvm-project//llvm:Support",
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
        ":type_to_shape",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
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
        "@llvm-project//llvm:Support",
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
        ":hlo_utils",
        "//tensorflow/compiler/mlir/hlo",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "attribute_importer",
    srcs = ["attribute_importer.cc"],
    hdrs = ["attribute_importer.h"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

tf_native_cc_binary(
    name = "operator_writer_gen",
    srcs = ["operator_writer_gen.cc"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
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
        "//tensorflow/compiler/mlir/hlo:hlo_ops_td_files",
        "//tensorflow/compiler/mlir/hlo:include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
    ],
    outs = ["operator_writers.inc"],
    cmd = ("$(location :operator_writer_gen) " +
           "-I external/llvm-project/mlir/include " +
           "-I tensorflow/compiler/mlir " +
           "-I tensorflow/compiler/mlir/hlo/include " +
           "$(location //tensorflow/compiler/mlir/hlo:include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td) " +
           " -o $@"),
    tools = [":operator_writer_gen"],
)

cc_library(
    name = "all_xla_passes_for_testing",
    visibility = [
        "//tensorflow/compiler/mlir:__subpackages__",
    ],
    deps = [
        ":chlo_legalize_to_hlo",
        ":hlo_legalize_to_lhlo",
        ":lhlo_copy_removal",
        ":lhlo_fuse_linalg",
        ":lhlo_legalize_to_affine",
        ":lhlo_legalize_to_gpu",
        ":lhlo_legalize_to_parallel_loops",
        ":xla_hlo_fusion",
        ":xla_hlo_to_lhlo_with_xla",
        ":xla_legalize_control_flow",
        ":xla_legalize_tanh_to_approximation",
        ":xla_legalize_tf",
        ":xla_legalize_tf_with_tf2xla",
        ":xla_legalize_to_linalg",
        ":xla_legalize_to_standard",
        ":xla_lower",
        ":xla_sink_constants_to_control_flow",
        ":xla_test_passes",
        ":xla_transform_unranked_hlo",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
        "//tensorflow/compiler/mlir/hlo:lhlo",
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
