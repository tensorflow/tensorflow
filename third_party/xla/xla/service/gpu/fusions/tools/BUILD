load("//xla:xla.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "mlir_fusions_opt",
    srcs = ["mlir_fusions_opt.cc"],
    # We want to use this tool for lit tests. Due to hermetic cuda, we need to
    # set linkopts in such a way that dynamic libraries are found, which are
    # symlinked from the lit_lib directory.
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    visibility = ["//xla/service/gpu/fusions:__subpackages__"],
    deps = [
        "//xla/backends/gpu/codegen/ir:xla_gpu",
        "//xla/backends/gpu/codegen/transforms:passes",
        "//xla/mlir_hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu/fusions/mlir:mlir_fusion_emitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "test_lib",
    testonly = 1,
    srcs = ["test_lib.cc"],
    hdrs = ["test_lib.h"],
    deps = [
        "//xla:status_macros",
        "//xla/backends/gpu/codegen/ir:xla_gpu",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu/fusions",
        "//xla/service/gpu/fusions/mlir:mlir_fusion_emitter",
        "//xla/stream_executor:device_description",
        "//xla/tools:hlo_module_loader",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

xla_cc_binary(
    name = "fusion_to_mlir",
    testonly = 1,
    srcs = ["fusion_to_mlir.cc"],
    # We want to use this tool for lit tests. Due to hermetic cuda, we need to
    # set linkopts in such a way that dynamic libraries are found, which are
    # symlinked from the lit_lib directory.
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    visibility = ["//xla/service/gpu/fusions:__subpackages__"],
    deps = [
        ":test_lib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_binary(
    name = "test_correctness",
    testonly = 1,
    srcs = ["test_correctness.cc"],
    # We want to use this tool for lit tests. Due to hermetic cuda, we need to
    # set linkopts in such a way that dynamic libraries are found, which are
    # symlinked from the lit_lib directory.
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    visibility = ["//xla/service/gpu/fusions:__subpackages__"],
    deps = [
        ":test_lib",
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)
