load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
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
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncTdFiles",
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
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_xla_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TfXla",
            ],
            "transforms/tf_xla_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_xla_passes.td",
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)

cc_library(
    name = "xla_passes",
    srcs = [
        "transforms/outline_with_xla_framework.cc",
        "transforms/xla_framework_to_llvm_pass.cc",
    ],
    hdrs = [
        "transforms/xla_passes.h",
    ],
    deps = [
        ":xla_framework",
        ":xla_passes_inc_gen",
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tf_xla_passes",
    srcs = [
        "transforms/xla_legalize_tf_passes.h.inc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":tf_xla_passes_inc_gen",
        ":xla_legalize_tf",
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "legalize_utils",
    srcs = ["transforms/utils.cc"],
    hdrs = ["transforms/utils.h"],
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "legalize_tf",
    srcs = [
        "transforms/generated_legalize_tf.inc",
        "transforms/legalize_tf.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":legalize_tf_patterns_inc_gen",
        ":legalize_utils",
        ":tf_xla_passes_inc_gen",
        ":xla_legalize_tf_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:dynamic_shape_utils",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/client/lib:conv_grad_size_util",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:convert_op_folder",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:attribute_importer",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:conv_grad_shape_utils",
        "//tensorflow/tsl/platform:bfloat16",
        "//tensorflow/tsl/platform:status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "adjust_layout",
    srcs = [
        "transforms/adjust_layout.cc",
    ],
    hdrs = [
        "transforms/adjust_layout.h",
    ],
    deps = [
        ":tf_xla_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/client/lib:conv_grad_size_util",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:convert_op_folder",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_conversions",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:conv_grad_shape_utils",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/tsl/platform:bfloat16",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "xla_legalize_tf",
    srcs = [
        "transforms/legalize_tf_collective.cc",
        "transforms/legalize_tf_communication.cc",
        "transforms/legalize_tf_control_flow.cc",
        "transforms/legalize_tf_types.cc",
        "transforms/tf_xla_passes.h.inc",
        "transforms/xla_legalize_tf.cc",
        "transforms/xla_legalize_tf_passes.h.inc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":legalize_tf",
        ":legalize_utils",
        ":xla_legalize_tf_passes_inc_gen",
        ":xla_legalize_tf_with_tf2xla",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:side_effect_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:chlo_legalize_to_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:convert_op_folder",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "xla_legalize_tf_no_fallback",
    srcs = [
        "transforms/xla_legalize_tf_no_fallback.cc",
        "transforms/xla_legalize_tf_passes.h.inc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":legalize_tf",
        ":tf_xla_passes_inc_gen",
        ":xla_legalize_tf_passes_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "xla_legalize_tf_with_tf2xla",
    srcs = [
        "transforms/legalize_tf_with_tf2xla.cc",
    ],
    deps = [
        ":tf_xla_passes_inc_gen",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
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
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/stream_executor:timer",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:mlir_hlo_builder",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "mhlo_to_lhlo_with_xla",
    srcs = ["transforms/mhlo_to_lhlo_with_xla.cc"],
    hdrs = ["transforms/mhlo_to_lhlo_with_xla.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc",
        "//tensorflow/compiler/xla/service/gpu:cublas_cudnn",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:attribute_importer",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_module_importer",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_utils",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "location_metadata",
    srcs = ["location_metadata.cc"],
    hdrs = ["location_metadata.h"],
    deps = [
        "//tensorflow/compiler/mlir:name_utils",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_binary(
    name = "xla-opt",
    testonly = True,
    srcs = ["xla_opt_main.cc"],
    deps = [
        ":adjust_layout",  # buildcleaner: keep
        ":mhlo_to_lhlo_with_xla",  # buildcleaner: keep
        ":tf_xla_passes",  # buildcleaner: keep
        ":xla_framework",
        ":xla_legalize_tf",  # buildcleaner: keep
        ":xla_legalize_tf_no_fallback",  # buildcleaner: keep
        ":xla_passes",  # buildcleaner: keep
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service/cpu:hlo_xla_runtime_pipeline",  # buildcleaner: keep
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:register",
    ],
)

tf_cc_binary(
    name = "xla-opt-gpu",
    testonly = True,
    srcs = ["xla_opt_main.cc"],
    deps = [
        ":adjust_layout",  # buildcleaner: keep
        ":mhlo_to_lhlo_with_xla",  # buildcleaner: keep
        ":tf_xla_passes",  # buildcleaner: keep
        ":xla_framework",  # buildcleaner: keep
        ":xla_legalize_tf",  # buildcleaner: keep
        ":xla_legalize_tf_no_fallback",  # buildcleaner: keep
        ":xla_passes",  # buildcleaner: keep
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:register",
    ],
)

td_library(
    name = "td_files",
    srcs = [
        "ir/xla_framework_ops.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "xla_framework_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/xla_framework.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/xla_framework.cc.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "ir/xla_framework_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "ir/xla_framework_dialect.cc.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "ir/xla_framework_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "ir/xla_framework_types.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/xla_framework_ops.td",
    deps = [":td_files"],
)

cc_library(
    name = "xla_framework",
    srcs = [
        "ir/xla_framework.cc",
        "ir/xla_framework.cc.inc",
        "ir/xla_framework.h.inc",
    ],
    hdrs = ["ir/xla_framework.h"],
    deps = [
        ":xla_framework_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
    ],
)
