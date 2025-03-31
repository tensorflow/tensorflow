load("//xla:py_strict.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//xla:xla.default.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "test_lib",
    testonly = 1,
    srcs = ["test_lib.cc"],
    hdrs = ["test_lib.h"],
    deps = [
        "//xla:status_macros",
        "//xla/backends/gpu/codegen:fusions",
        "//xla/backends/gpu/codegen/emitters:emitter_base",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:hlo_fusion_analysis",
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
    visibility = ["//xla/backends/gpu/codegen:__subpackages__"],
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
    visibility = ["//xla/backends/gpu/codegen:__subpackages__"],
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

xla_cc_binary(
    name = "fusion_wrapper",
    testonly = 1,
    srcs = ["fusion_wrapper.cc"],
    visibility = ["//xla/backends/gpu/codegen:__subpackages__"],
    deps = [
        ":test_lib",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

py_strict_binary(
    name = "ncu_rep",
    srcs = ["ncu_rep.py"],
    deps = [
        ":ncu_rep_lib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_library(
    name = "ncu_rep_lib",
    srcs = ["ncu_rep_lib.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_test(
    name = "ncu_rep_test",
    srcs = ["ncu_rep_test.py"],
    deps = [
        ":ncu_rep_lib",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)
