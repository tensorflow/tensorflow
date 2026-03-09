load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")
load("//xla:xla.default.bzl", "xla_cc_binary")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

package_group(
    name = "codegen_tests",
    packages = [
        "//xla/backends/gpu/codegen/emitters/tests",
        "//xla/codegen/emitters/tests",
    ],
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
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
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
    visibility = [":codegen_tests"],
    deps = [
        ":test_lib",
        "//xla/codegen/tools:test_lib",
        "//xla/hlo/analysis:symbolic_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_binary(
    name = "gpu_test_correctness",
    testonly = 1,
    srcs = ["gpu_test_correctness.cc"],
    # We want to use this tool for lit tests. Due to hermetic cuda, we need to
    # set linkopts in such a way that dynamic libraries are found, which are
    # symlinked from the lit_lib directory.
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    visibility = [":codegen_tests"],
    deps = [
        ":test_lib",
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla:shape_util",
        "//xla/codegen/tools:test_lib",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/analysis:indexing_test_utils",
        "//xla/hlo/analysis:symbolic_map",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:pjrt_gpu_client_registry",
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
        "@tsl//tsl/platform:statusor",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:all_runtime",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:all_runtime",
    ]),
)

xla_cc_binary(
    name = "fusion_wrapper",
    testonly = 1,
    srcs = ["fusion_wrapper.cc"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla/codegen/tools:test_lib",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:platform_port",
    ],
)

py_binary(
    name = "ncu_rep",
    srcs = ["ncu_rep.py"],
    strict_deps = True,
    deps = [
        ":ncu_rep_lib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "ncu_rep_lib",
    srcs = ["ncu_rep_lib.py"],
    strict_deps = True,
    deps = ["@absl_py//absl:app"],
)

py_test(
    name = "ncu_rep_test",
    srcs = ["ncu_rep_test.py"],
    strict_deps = True,
    deps = [
        ":ncu_rep_lib",
        "@absl_py//absl/testing:absltest",
    ],
)
