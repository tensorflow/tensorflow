# Placeholder: load py_proto_library
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//learning/brain/mlir/quantization/stablehlo/python/integration_test/...",
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/compiler/mlir/quantization/...",
        "//tensorflow/compiler/mlir/tf2xla/transforms/...",
        "//tensorflow/lite/...",
        "//third_party/cloud_tpu/inference_converter/...",  # TPU Inference Converter V1
        "//waymo/ml/compiler/mlir/...",
    ] + internal_visibility_allowlist(),
)

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        ":internal_visibility_allowlist_package",
        "//tensorflow:__pkg__",
    ],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "stablehlo_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/passes.h.inc": [
        "-gen-pass-decls",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "passes/convert_func_to_bfloat16.cc",
        "passes/convert_shape_constraint_to_assert.cc",
        "passes/convert_xla_call_module_op_to_bfloat16.cc",
        "passes/defer_activation_transpose.cc",
        "passes/fold_constant_transpose.cc",
        "passes/insert_calibration_statistics_saver.cc",
        "passes/insert_weight_param.cc",
        "passes/lift_quantizable_spots_as_functions.cc",
        "passes/lift_quantizable_spots_as_functions_fusion.inc",
        "passes/lift_quantizable_spots_as_functions_simple.inc",
        "passes/merge_fusion_with_dequantize.cc",
        "passes/nchw_convolution_to_nhwc.cc",
        "passes/optimize_graph.cc",
        "passes/post_quantize.cc",
        "passes/prepare_quantize.cc",
        "passes/quantize.cc",
        "passes/quantize_composite_functions.cc",
        "passes/quantize_weight.cc",
        "passes/remove_sharding_custom_call.cc",
        "passes/remove_sharding_custom_call.inc",
        "passes/replace_stablehlo_ops_in_main_function_with_xla_call_module_ops.cc",
        "passes/restore_function_name.cc",
        "passes/unfuse_mhlo_batch_norm.cc",
        "passes/unwrap_xla_call_module_op.cc",
        "passes/xla_call_module_to_call.cc",
    ],
    hdrs = [
        "passes/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":bfloat16_type",
        ":fill_quantization_options",
        ":lift_quantizable_spots_as_functions_fusion_inc_gen",
        ":lift_quantizable_spots_as_functions_simple_inc_gen",
        ":optimize_graph_inc_gen",
        ":quantization_config_proto_cc",
        ":quantization_options_proto_cc",
        ":quantization_patterns",
        ":remove_sharding_custom_call_inc_gen",
        ":stablehlo_passes_inc_gen",
        ":stablehlo_type_utils",
        "//tensorflow/compiler/mlir/quantization/common:func",
        "//tensorflow/compiler/mlir/quantization/common:lift_as_function_call",
        "//tensorflow/compiler/mlir/quantization/common:tf_attrs_and_constraints",
        "//tensorflow/compiler/mlir/quantization/common:uniform_quantized_types",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:permutation",
        "//tensorflow/compiler/mlir/quantization/stablehlo/ops:stablehlo_op_quant_spec",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_quant_ops",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:xla_call_module_attrs",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/ir/types:Dialect",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:regexp",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_portable_api",
        "@stablehlo//:stablehlo_serialization",
        "@stablehlo//:version",
    ],
)

cc_library(
    name = "quantization_patterns",
    srcs = ["passes/quantization_patterns.cc"],
    hdrs = [
        "passes/quantization_patterns.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/common:lift_as_function_call",
        "//tensorflow/compiler/mlir/quantization/common:tf_attrs_and_constraints",
        "//tensorflow/compiler/mlir/quantization/common:uniform_quantized_types",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/stablehlo/ops:stablehlo_op_quant_spec",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

td_library(
    name = "quant_td_files",
    srcs = [
        "passes/lift_quantizable_spots_as_functions_fusion.td",
        "passes/lift_quantizable_spots_as_functions_simple.td",
        "passes/optimize_graph.td",
        "passes/remove_sharding_custom_call.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "//tensorflow/compiler/mlir/quantization/common:lift_as_function_call_td_files",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quant_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@stablehlo//:stablehlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "lift_quantizable_spots_as_functions_simple_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/lift_quantizable_spots_as_functions_simple.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/lift_quantizable_spots_as_functions_simple.td",
    deps = [
        ":quant_td_files",
        "//tensorflow/compiler/mlir/quantization/common:quant_td_files",
    ],
)

gentbl_cc_library(
    name = "lift_quantizable_spots_as_functions_fusion_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/lift_quantizable_spots_as_functions_fusion.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/lift_quantizable_spots_as_functions_fusion.td",
    deps = [
        ":quant_td_files",
        "//tensorflow/compiler/mlir/quantization/common:quant_td_files",
    ],
)

gentbl_cc_library(
    name = "optimize_graph_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/optimize_graph.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/optimize_graph.td",
    deps = [
        ":quant_td_files",
        "//tensorflow/compiler/mlir/quantization/common:quant_td_files",
    ],
)

gentbl_cc_library(
    name = "remove_sharding_custom_call_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/remove_sharding_custom_call.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/remove_sharding_custom_call.td",
    deps = [
        ":quant_td_files",
    ],
)

gentbl_cc_library(
    name = "bridge_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/bridge/passes.h.inc": [
        "-gen-pass-decls",
        "-name=Bridge",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/bridge/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "bridge_passes",
    srcs = [
        "passes/bridge/convert_tf_quant_ops_to_mhlo.cc",
        "passes/bridge/convert_tf_quant_types.cc",
        "passes/bridge/optimize.cc",
        "passes/bridge/passes.cc",
        "passes/bridge/verify_quant_legalization.cc",
    ],
    hdrs = [
        "passes/bridge/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        ":internal_visibility_allowlist_package",
        "//tensorflow/compiler/mlir:__subpackages__",
    ],
    deps = [
        ":bridge_passes_inc_gen",
        ":optimize_inc_gen",
        ":tf_type_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tf2xla/transforms:legalize_utils",
        "//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_targets",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/util/quantization:uniform_quant_ops_attr_proto_cc",
        "//tensorflow/core/util/quantization:uniform_quant_ops_params",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/translate/hlo_to_mhlo:attribute_importer",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_passes",
    ],
)

td_library(
    name = "optimize_td_files",
    srcs = [
        "passes/bridge/optimize.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@local_xla//xla/mlir_hlo:hlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "optimize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/bridge/optimize.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/bridge/optimize.td",
    deps = [":optimize_td_files"],
)

tf_cc_test(
    name = "convert_tf_quant_to_mhlo_int_test",
    srcs = [
        "passes/bridge/convert_tf_quant_to_mhlo_int_test.cc",
    ],
    tags = ["nomac"],  # TODO(b/297362678): re-enable mac test.
    deps = [
        ":bridge_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:constant_fold",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:nn",
        "//tensorflow/core/kernels/uniform_quant_ops:kernels",
        "//tensorflow/core/ops",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@local_xla//xla:error_spec",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:pjrt_executable",
        "@local_xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@local_xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@local_xla//xla/tests:literal_test_util",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
        "@stablehlo//:chlo_ops",
    ],
)

tf_cc_test(
    name = "legalize_tf_quant_test",
    srcs = ["passes/bridge/legalize_tf_quant_test.cc"],
    # TODO(b/299290907): Fix OSS test.
    tags = ["no_oss"],
    deps = [
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:legalize_tf",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
)

tf_cc_test(
    name = "convert_tf_quant_types_test",
    srcs = ["passes/bridge/convert_tf_quant_types_test.cc"],
    deps = [
        ":bridge_passes",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/core:test",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "quantize_passes",
    srcs = [
        "quantize_passes.cc",
    ],
    hdrs = [
        "quantize_passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [":internal_visibility_allowlist_package"],
    deps = [
        ":fill_quantization_options",
        ":passes",
        ":quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/core/platform:path",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

gentbl_cc_library(
    name = "stablehlo_test_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes/testing/passes.h.inc": [
        "-gen-pass-decls",
        "-name=Test",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/testing/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "test_passes",
    srcs = [
        "passes/testing/test_lift_quantizable_spots_as_functions_with_quantization_specs.cc",
        "passes/testing/test_post_calibration_component.cc",
        "passes/testing/test_pre_calibration_component.cc",
        "passes/testing/test_tf_to_stablehlo_pass.cc",
    ],
    hdrs = [
        "passes/testing/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":passes",
        ":quantization_config_proto_cc",
        ":stablehlo_test_passes_inc_gen",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:config",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:post_calibration",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:pre_calibration",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:UBDialect",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/mlir_hlo",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
    ],
)

cc_library(
    name = "fill_quantization_options",
    srcs = ["utils/fill_quantization_options.cc"],
    hdrs = ["utils/fill_quantization_options.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/core/platform:path",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "math_utils",
    srcs = ["utils/math_utils.cc"],
    hdrs = ["utils/math_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["@llvm-project//mlir:Support"],
)

tf_cc_test(
    name = "math_utils_test",
    srcs = ["utils/math_utils_test.cc"],
    deps = [
        ":math_utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "stablehlo_type_utils",
    hdrs = ["utils/stablehlo_type_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/platform:status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

tf_cc_test(
    name = "stablehlo_type_utils_test",
    srcs = ["utils/stablehlo_type_utils_test.cc"],
    deps = [
        ":stablehlo_type_utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "tf_type_utils",
    srcs = ["utils/tf_type_utils.cc"],
    hdrs = ["utils/tf_type_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:numeric_types",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

tf_cc_test(
    name = "tf_type_utils_test",
    srcs = [
        "utils/tf_type_utils_test.cc",
    ],
    deps = [
        ":bridge_passes",
        ":tf_type_utils",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:path",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/tsl/framework:numeric_types",
    ],
)

cc_library(
    name = "bfloat16_type",
    srcs = ["utils/bfloat16_type.cc"],
    hdrs = ["utils/bfloat16_type.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/platform:status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

tf_cc_test(
    name = "bfloat16_type_test",
    srcs = [
        "utils/bfloat16_type_test.cc",
    ],
    deps = [
        ":bfloat16_type",
        "//tensorflow/compiler/mlir:register_common_dialects",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:path",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/tsl/framework:numeric_types",
    ],
)

# OSS only: This target is header-only. Link `quantization_options_proto_cc_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link
# `quantization_options_proto_cc_impl` directly unless the target does not link
# `libtensorflow_framework.so`.
tf_proto_library(
    name = "quantization_options_proto",
    srcs = ["quantization_options.proto"],
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "quantization_options_py_pb2",
#     visibility = [":internal_visibility_allowlist_package"],
#     deps = [":quantization_options_proto"],
# )
# copybara:uncomment_end

# OSS only: This target is header-only. Link `quantization_config_proto_cc_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link
# `quantization_config_proto_cc_impl` directly unless the target does not link
# `libtensorflow_framework.so`.
tf_proto_library(
    name = "quantization_config_proto",
    srcs = ["quantization_config.proto"],
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "quantization_config_py_pb2",
#     visibility = [
#         ":internal_visibility_allowlist_package",
#     ],
#     deps = [":quantization_config_proto"],
# )
# copybara:uncomment_end

exports_files([
    "run_lit.sh",
])

tf_cc_binary(
    name = "stablehlo-quant-opt",
    srcs = ["tools/stablehlo_quant_opt.cc"],
    visibility = [":internal_visibility_allowlist_package"],
    deps = [
        ":bridge_passes",
        ":passes",
        ":test_passes",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:pass_pipeline",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:vhlo_ops",
    ],
)
