# Tools for Triton GPU backend.

load("//xla:xla.default.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla/backends/gpu/codegen:__subpackages__",
    ],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "hlo_to_xtileir",
    srcs = ["hlo_to_xtileir.cc"],
    # We want to use this tool for lit tests. Due to hermetic cuda, we need to
    # set linkopts in such a way that dynamic libraries are found, which are
    # symlinked from the lit_lib directory.
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/codegen/triton:xtile_compiler",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu/model:block_level_parameters",
        "//xla/tools:hlo_module_loader",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@tsl//tsl/platform:platform_port",
    ],
)
