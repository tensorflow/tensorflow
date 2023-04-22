load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_native_cc_binary")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["@llvm-project//mlir:subpackages"],
    packages = [
        "//babelfish/device/...",
        "//learning/brain/experimental/dtensor/...",
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/google/xla/kernels/...",
        "//learning/brain/google/xla/mlir/...",
        "//learning/deepmind/partir/...",
        "//learning/pathways/data_parallel/tf2xla/...",
        "//platforms/darwinn/contrib/waymo/...",
        "//platforms/xla/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
    ],
)

gentbl_cc_library(
    name = "legalize_tf_patterns_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_legalize_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_tf_patterns.td",
    deps = [
        "//tensorflow/compiler/mlir/hlo:hlo_ops_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "xla_legalize_tf_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=LegalizeTf",
            ],
            "transforms/xla_legalize_tf_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/xla_legalize_tf_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "xla_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Xla",
            ],
            "transforms/xla_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/xla_passes.td",
    deps = [
        "//tensorflow/compiler/mlir/hlo:hlo_ops_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

cc_library(
    name = "xla_passes",
    srcs = [
        "transforms/prepare_for_export.cc",
        "transforms/xla_legalize_tf_passes.h.inc",
        "transforms/xla_passes.h.inc",
        "transforms/xla_passes_detail.h",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":xla_passes_inc_gen",
        "//tensorflow/compiler/mlir/hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_legalize_tf",
    srcs = [
        "transforms/generated_legalize_tf.inc",
        "transforms/legalize_tf.cc",
        "transforms/legalize_tf_communication.cc",
        "transforms/legalize_tf_control_flow.cc",
        "transforms/legalize_tf_types.cc",
        "transforms/xla_legalize_tf_passes.h.inc",
        "transforms/xla_legalize_tf_passes_detail.h",
        "transforms/xla_passes.h.inc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":attribute_importer",
        ":legalize_tf_patterns_inc_gen",
        ":type_to_shape",
        ":xla_legalize_tf_passes_inc_gen",
        ":xla_legalize_tf_with_tf2xla",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:chlo_legalize_to_hlo",
        "//tensorflow/compiler/mlir/hlo:convert_op_folder",
        "//tensorflow/compiler/mlir/hlo:hlo_ops_base_structs",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/client/lib:conv_grad_size_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:conv_grad_shape_utils",
        "//tensorflow/core/platform:bfloat16",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/stream_executor/tpu:c_api_conversions",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
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
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/tf2xla:xla_compilation_device",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_expression",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
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
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mhlo_to_lhlo_with_xla",
    srcs = ["transforms/mhlo_to_lhlo_with_xla.cc"],
    hdrs = ["transforms/mhlo_to_lhlo_with_xla.h"],
    deps = [
        ":attribute_importer",
        ":hlo_module_importer",
        ":hlo_utils",
        ":mlir_hlo_to_hlo",
        ":translate_cl_options",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:hlo_ops_base_enums",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_casting_utils",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Translation",
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
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "hlo_utils_test",
    srcs = ["hlo_utils_test.cc"],
    deps = [
        ":hlo_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
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
        ":hlo_utils",
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
        ":attribute_exporter",
        ":type_to_shape",
        ":xla_passes",
        "//tensorflow/compiler/mlir:name_utils",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_helpers",
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
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
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
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_casting_utils",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
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
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core/platform:types",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "attribute_exporter",
    srcs = ["attribute_exporter.cc"],
    hdrs = ["attribute_exporter.h"],
    deps = [
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor:dnn",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "translate_cl_options",
    srcs = ["xla_mlir_translate_cl.cc"],
    hdrs = ["xla_mlir_translate_cl.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_mlir_translate",
    srcs = ["xla_mlir_translate.cc"],
    hdrs = ["xla_mlir_translate.h"],
    deps = [
        ":hlo_to_mlir_hlo",
        ":mhlo_to_lhlo_with_xla",
        ":mlir_hlo_to_hlo",
        ":translate_cl_options",
        ":type_to_shape",
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TensorDialect",
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

gentbl_cc_library(
    name = "operator_writer_inc",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [([], "operator_writers.inc")],
    tblgen = ":operator_writer_gen",
    td_file = "//tensorflow/compiler/mlir/hlo:include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td",
    deps = [
        "//tensorflow/compiler/mlir/hlo:hlo_ops_td_files",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

cc_library(
    name = "all_xla_passes_for_testing",
    visibility = [
        "//tensorflow/compiler/mlir:__subpackages__",
    ],
    deps = [
        ":mhlo_to_lhlo_with_xla",
        ":xla_legalize_tf",
        ":xla_legalize_tf_with_tf2xla",
        ":xla_passes",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:chlo_legalize_to_hlo",
        "//tensorflow/compiler/mlir/hlo:hlo_legalize_to_lhlo",
        "//tensorflow/compiler/mlir/hlo:legalize_control_flow",
        "//tensorflow/compiler/mlir/hlo:legalize_to_linalg",
        "//tensorflow/compiler/mlir/hlo:legalize_to_standard",
        "//tensorflow/compiler/mlir/hlo:legalize_trigonometric_to_approximation",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_fuse_linalg",
        "//tensorflow/compiler/mlir/hlo:lhlo_legalize_to_affine",
        "//tensorflow/compiler/mlir/hlo:lhlo_legalize_to_gpu",
        "//tensorflow/compiler/mlir/hlo:lhlo_legalize_to_parallel_loops",
        "//tensorflow/compiler/mlir/hlo:mhlo_fusion",
        "//tensorflow/compiler/mlir/hlo:mhlo_to_mhlo_lowering_patterns",
        "//tensorflow/compiler/mlir/hlo:sink_constants_to_control_flow",
        "//tensorflow/compiler/mlir/hlo:test_passes",
    ],
)

tf_cc_binary(
    name = "xla-opt",
    deps = [
        ":all_xla_passes_for_testing",
        "//tensorflow/compiler/mlir:tf_mlir_opt_main",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:gpu_plugin",
    ],
)
