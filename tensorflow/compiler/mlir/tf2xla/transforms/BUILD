# Description:
#    TF2XLA Bridge transforms

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tf2xla:tensorflow_mlir_tf2xla",
    ],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "legalize_tf_patterns_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "generated_legalize_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "legalize_tf_patterns.td",
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
            "xla_legalize_tf_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "xla_legalize_tf_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
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
            "tf_xla_passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_xla_passes.td",
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
    name = "tf_xla_passes",
    srcs = [
        "xla_legalize_tf_passes.h.inc",
    ],
    hdrs = [
        "passes.h",
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
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "legalize_utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/platform:errors",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "legalize_tf",
    srcs = [
        "generated_legalize_tf.inc",
        "legalize_tf.cc",
    ],
    hdrs = [
        "passes.h",
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
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "xla_legalize_targets",
    srcs = [
        "xla_legalize_targets.cc",
    ],
    hdrs = [
        "xla_legalize_targets.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla/mlir_hlo",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:chlo_ops",
    ],
)

tf_cc_test(
    name = "xla_legalize_targets_test",
    srcs = ["xla_legalize_targets_test.cc"],
    deps = [
        ":xla_legalize_targets",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:chlo_ops",
    ],
)

tf_cc_test(
    name = "verify_tfxla_legalization_test",
    srcs = ["verify_tfxla_legalization_test.cc"],
    deps = [
        ":legalize_tf",
        ":test_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "xla_legalize_tf",
    srcs = [
        "convert_mhlo_quant_to_int.cc",
        "infeed_ops_xla_adjust_layout.cc",
        "legalize_tf_collective.cc",
        "legalize_tf_communication.cc",
        "legalize_tf_types.cc",
        "tf_xla_passes.h.inc",
        "tfxla_device_specific_transforms.cc",
        "verify_tfxla_legalization.cc",
        "xla_legalize_tf.cc",
        "xla_legalize_tf_passes.h.inc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":legalize_tf",
        ":legalize_utils",
        ":xla_legalize_targets",
        ":xla_legalize_tf_no_fallback",
        ":xla_legalize_tf_passes_inc_gen",
        ":xla_legalize_tf_with_tf2xla",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:set_tpu_infeed_layout",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/tf2xla/kernels:rng_converter_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:side_effect_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:chlo_legalize_to_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:convert_op_folder",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_conversions",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_api",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:attribute_importer",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util/quantization:uniform_quant_ops_params",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
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
        "xla_legalize_tf_no_fallback.cc",
        "xla_legalize_tf_passes.h.inc",
    ],
    hdrs = [
        "passes.h",
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
    name = "tf2xla_rewriter",
    srcs = [
        "tf2xla_rewriter.cc",
    ],
    hdrs = [
        "tf2xla_rewriter.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":legalize_tf",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:export_tf_dialect_op",
        "//tensorflow/compiler/mlir/tensorflow:tpu_embedding_ops_registry",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/tf2xla:xla_compilation_device",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_expression",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_module_importer",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:mlir_hlo_builder",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
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

tf_cc_test(
    name = "tf2xla_rewriter_test",
    srcs = [
        "tf2xla_rewriter_test.cc",
    ],
    deps = [
        ":test_utils",
        ":tf2xla_rewriter",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "xla_legalize_tf_with_tf2xla",
    srcs = [
        "legalize_tf_with_tf2xla.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":tf2xla_rewriter",
        ":tf_xla_passes_inc_gen",
        ":xla_legalize_tf_passes_inc_gen",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:export_tf_dialect_op",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tpu_embedding_ops_registry",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/tf2xla:xla_compilation_device",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_expression",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/stream_executor:timer",
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

tf_cc_test(
    name = "xla_legalize_tf_test",
    srcs = [
        "xla_legalize_tf_test.cc",
    ],
    deps = [
        ":tf_xla_passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/tf2xla:xla_compilation_device",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_expression",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)
