load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load("//xla:lit.bzl", "lit_device_test")
load("//xla:xla.default.bzl", "xla_cc_binary")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "triton_test_correctness",
    testonly = 1,
    srcs = ["triton_test_correctness.cc"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla/backends/gpu/codegen/tools:test_lib",
        "//xla/codegen/tools:test_lib",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:pjrt_gpu_client_registry",
        "//xla/tools:hlo_module_loader",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:all_runtime",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor/rocm:all_runtime",
    ]) + if_sycl_is_configured([
        "//xla/stream_executor/sycl:all_runtime",
    ]),
)

lit_device_test(
    name = "tests",
    srcs = glob(["**/*.hlo"]),
    backends = [
        "h100",
        "b200",
    ],
    tools = [
        ":triton_test_correctness",
        "//xla/backends/gpu/codegen/triton/tools:hlo_to_xtileir",
        "@llvm-project//llvm:FileCheck",
    ],
)
