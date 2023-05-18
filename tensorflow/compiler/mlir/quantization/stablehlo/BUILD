load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

# TODO(b/264218457): Create stablehlo-quantization-opt and register passes to actually test.

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/compiler/mlir/quantization/...",
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
        "passes/quantize_weight.cc",
    ],
    hdrs = [
        "passes/passes.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":fill_quantization_options",
        ":quantization_options_proto_cc",
        ":stablehlo_passes_inc_gen",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/core/platform:path",
        "//third_party/eigen3",
        "@llvm-project//llvm:Support",
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

cc_library(
    name = "quantize_passes",
    srcs = [
        "quantize_passes.cc",
    ],
    hdrs = [
        "quantize_passes.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    visibility = [":internal_visibility_allowlist_package"],
    deps = [
        ":fill_quantization_options",
        ":passes",
        ":quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

gentbl_cc_library(
    name = "stablehlo_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
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
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":quantization_options_proto_cc",
        "@llvm-project//llvm:Support",
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
