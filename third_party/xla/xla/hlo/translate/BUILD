load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xla:xla.bzl", "xla_cc_binary")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:tensorflow_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "portable_api",
    srcs = ["portable_api.cc"],
    hdrs = ["portable_api.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":stablehlo",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
        "@stablehlo//:register",
    ],
)

build_test(
    name = "xla-translate_build_test",
    targets = [
        ":xla-translate",
    ],
)

xla_cc_binary(
    name = "xla-translate",
    testonly = True,
    srcs = ["xla_translate_main.cc"],
    deps = [
        "//xla/hlo/translate/hlo_to_mhlo:translate_registration",
        "//xla/hlo/translate/mhlo_to_hlo:translate_registration",
        "//xla/hlo/translate/stablehlo_to_hlo:translate_registration",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_transfer_manager",
        "//xla/stream_executor/host:host_platform",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

build_test(
    name = "xla-translate-opt_build_test",
    targets = [
        ":xla-translate-opt",
    ],
)

xla_cc_binary(
    name = "xla-translate-opt",
    testonly = True,
    srcs = ["xla_translate_opt_main.cc"],
    deps = [
        "//xla/mlir/framework/ir:xla_framework",
        "//xla/mlir/framework/transforms:passes",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/service:cpu_plugin",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@local_tsl//tsl/platform:platform_port",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "stablehlo",
    srcs = ["stablehlo.cc"],
    hdrs = ["stablehlo.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:debug_options_flags",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_module_importer",
        "//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//xla/hlo/translate/mhlo_to_hlo:module_attributes_exporter",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service:hlo_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBDialect",
        "@local_tsl//tsl/platform:errors",
        "@stablehlo//:register",
    ],
)
