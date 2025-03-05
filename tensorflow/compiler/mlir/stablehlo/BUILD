load("@local_xla//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/mlir/lite/stablehlo/...",
        "//tensorflow/compiler/mlir/quantization/tensorflow/...",
        "//tensorflow/compiler/tests/...",
    ],
)

tsl_pybind_extension(
    name = "stablehlo_extension",
    srcs = [
        "stablehlo.cc",
        "@stablehlo//:stablehlo/integrations/python/StablehloApi.cpp",
    ],
    hdrs = [
        "@stablehlo//:stablehlo/integrations/python/StablehloApi.h",
    ],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//third_party/python_runtime:headers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonHeadersAndDeps",
        "@nanobind",
        "@stablehlo//:stablehlo_capi",
    ],
)

pytype_strict_library(
    name = "stablehlo",
    srcs = ["stablehlo.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":stablehlo_extension",
    ],
)

py_strict_test(
    name = "stablehlo_test",
    srcs = ["stablehlo_test.py"],
    deps = [
        ":stablehlo",
        #internal proto upb dep
    ],
)

cc_library(
    name = "fold_broadcast_pass",
    srcs = [
        "transforms/fold_broadcast_pass.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "unfuse_batch_norm_pass",
    srcs = [
        "transforms/mhlo_passes/unfuse_batch_norm_pass.cc",
    ],
    hdrs = [
        "transforms/stablehlo_passes.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = [
        "-Ithird_party",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
    alwayslink = 1,
)
