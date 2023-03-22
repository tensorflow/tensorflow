load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "mhlo_to_lhlo_with_xla",
    srcs = ["mhlo_to_lhlo_with_xla.cc"],
    hdrs = ["mhlo_to_lhlo_with_xla.h"],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir/utils:error_util",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/gpu:backend_configs_cc",
        "//tensorflow/compiler/xla/service/gpu:cublas_cudnn",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/gpu:matmul_utils",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:attribute_importer",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_module_importer",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_utils",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TranslateLib",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    srcs = ["translate_registration.cc"],
    deps = [
        ":mhlo_to_lhlo_with_xla",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
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
        ":mhlo_to_lhlo_with_xla",  # buildcleaner: keep
        "//tensorflow/compiler/xla/mlir/framework/ir:xla_framework",
        "//tensorflow/compiler/xla/mlir/framework/transforms:passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service/cpu:hlo_xla_runtime_pipeline",  # buildcleaner: keep
        "//tensorflow/tsl/platform:platform_port",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:register",
    ],
)

build_test(
    name = "xla-translate-gpu-opt_build_test",
    targets = [
        ":xla-translate-gpu-opt",
    ],
)

xla_cc_binary(
    name = "xla-translate-gpu-opt",
    testonly = True,
    srcs = ["xla_translate_opt_main.cc"],
    deps = [
        ":mhlo_to_lhlo_with_xla",  # buildcleaner: keep
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:register",
        "//tensorflow/compiler/xla/mlir/framework/ir:xla_framework",
        "//tensorflow/compiler/xla/mlir/framework/transforms:passes",
        "//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_impl",
        "//tensorflow/tsl/platform:platform_port",
    ] + if_cuda(["//tensorflow/compiler/xla/stream_executor/cuda:cublas_plugin"]),
)
