# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

# TODO(b/264218457): Create stablehlo-quantization-opt and register passes to actually test.

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/compiler/mlir/quantization/...",
        "//tensorflow/compiler/mlir/tf2xla/transforms/...",
        "//tensorflow/lite/...",
        "//third_party/cloud_tpu/inference_converter/...",  # TPU Inference Converter V1
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

# TODO(b/264218457): Add quantize and post_quantize passes.
cc_library(
    name = "passes",
    srcs = [
        "passes/prepare_srq_quantize.cc",
        "passes/prepare_srq_quantize.inc",
        "passes/quantize_weight.cc",
    ],
    hdrs = [
        "passes/passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":fill_quantization_options",
        ":quantization_options_proto_cc",
        ":stablehlo_passes_inc_gen",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/core/platform:path",
        "//third_party/eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
    # Alwayslink is required for registering the MLIR passes.
    # TODO(b/255530126): Split the pass registration from the definitions to avoid binary size bloat.
    alwayslink = True,
)

td_library(
    name = "quant_td_files",
    srcs = [
        "passes/prepare_srq_quantize.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@stablehlo//:stablehlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "prepare_srq_quantize_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "passes/prepare_srq_quantize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/prepare_srq_quantize.td",
    deps = [":quant_td_files"],
)

gentbl_cc_library(
    name = "bridge_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Bridge",
            ],
            "passes/bridge/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/bridge/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "bridge_passes",
    srcs = [
        "passes/bridge/convert_mhlo_quant_to_int.cc",
        "passes/bridge/convert_tf_quant_ops_to_mhlo.cc",
        "passes/bridge/convert_tf_quant_types.cc",
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
        ":math_utils",
        ":tf_type_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mangling_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tf2xla/transforms:legalize_utils",
        "//tensorflow/compiler/mlir/tf2xla/transforms:xla_legalize_targets",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:chlo_legalize_to_hlo",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:attribute_importer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/util/quantization:uniform_quant_ops_attr_proto_cc",
        "//tensorflow/core/util/quantization:uniform_quant_ops_params",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:chlo_ops",
    ],
    # Force link to ensure ConvertTFQuantOpsToMHLOPass is registered.
    alwayslink = True,
)

tf_cc_test(
    name = "convert_tf_quant_to_mhlo_int_test",
    srcs = [
        "passes/bridge/convert_tf_quant_to_mhlo_int_test.cc",
    ],
    deps = [
        ":bridge_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@stablehlo//:chlo_ops",
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
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
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
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

gentbl_cc_library(
    name = "stablehlo_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
            ],
            "passes/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes/passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
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
        "//tensorflow/core/platform:status",
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
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:path",
        "//tensorflow/tsl/framework:numeric_types",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
    ],
)

tf_proto_library(
    name = "quantization_options_proto",
    srcs = ["quantization_options.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "quantization_options_py_pb2",
#     api_version = 2,
#     visibility = [":internal_visibility_allowlist_package"],
#     deps = [":quantization_options_proto"],
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
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "uniform_quantized_types",
    srcs = ["uniform_quantized_types.cc"],
    hdrs = ["uniform_quantized_types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "uniform_quantized_types_test",
    srcs = ["uniform_quantized_types_test.cc"],
    deps = [
        ":uniform_quantized_types",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)
