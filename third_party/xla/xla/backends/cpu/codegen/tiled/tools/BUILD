# Tools for Tiled CPU backend.

load("//xla:xla.default.bzl", "xla_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla/backends/gpu/codegen:__subpackages__",
    ],
    licenses = ["notice"],
)

xla_cc_binary(
    name = "fusion_to_xtile",
    srcs = ["fusion_to_xtile.cc"],
    visibility = [
        "//xla/backends/cpu/codegen/tiled:__subpackages__",
        "//xla/backends/gpu/codegen:__subpackages__",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu/codegen:fusion_compiler",
        "//xla/backends/cpu/codegen/tiled:tiled_fusion_emitter",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu/model:block_level_parameters",
        "//xla/tools:hlo_module_loader",
        "//xla/tsl/platform:logging",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_binary(
    name = "test_correctness",
    testonly = 1,
    srcs = ["test_correctness.cc"],
    visibility = [
        "//xla/backends/cpu/codegen/tiled:__subpackages__",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla/hlo/ir:hlo",
        "//xla/service:cpu_plugin",
        "//xla/tests:hlo_interpreter_reference_mixin",
        "//xla/tests:hlo_test_base",
        "//xla/tests:pjrt_cpu_client_registry",
        "//xla/tools:hlo_module_loader",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
    ],
)
