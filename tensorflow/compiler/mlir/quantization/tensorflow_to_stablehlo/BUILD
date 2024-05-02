load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.default.bzl", "filegroup", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//learning/brain/mlir/quantization/stablehlo/python/integration_test/...",
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

cc_library(
    name = "tf_to_stablehlo",
    srcs = [
        "tf_to_stablehlo.cc",
    ],
    hdrs = [
        "tf_to_stablehlo.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:saved_model_import",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "//tensorflow/compiler/mlir/tensorflow/transforms:shape_inference_pass",
        "//tensorflow/core:core_cpu_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

tf_cc_binary(
    name = "tf-to-stablehlo-translate",
    srcs = [
        "tf_to_stablehlo_translate.cc",
    ],
    visibility = [":internal_visibility_allowlist_package"],
    deps = [
        ":tf_to_stablehlo",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    # TODO: b/288344501 - Enable OSS tests again when stable-quant-opt works well.
    default_tags = [
        "no_oss",
        "no_pip",
    ],
    driver = "//tensorflow/compiler/mlir/quantization/tensorflow_to_stablehlo:run_lit.sh",
    size_override = {
    },
    tags_override = {
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tf-to-stablehlo-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
