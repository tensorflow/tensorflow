load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

xla_test(
    name = "cpu_aot_compiler_test",
    srcs = ["cpu_aot_compiler_test.cc"],
    backends = [
        "cpu",
    ],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/backends/cpu:target_machine_options",
        "//xla/backends/cpu/codegen:cpu_features",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:hlo_runner_interface",
        "//xla/service/cpu:cpu_aot_compilation_result",
        "//xla/service/cpu:test_header_helper",
        "//xla/service/restricted:hlo_runner",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@tsl//tsl/platform:casts",
    ],
)

xla_cc_test(
    name = "cpu_compiler_internals_test",
    srcs = ["cpu_compiler_internals_test.cc"],
    tags = ["pjrt_migration_candidate"],
    deps = [
        "//xla/backends/cpu/codegen/emitters:cpu_fusion_emitter_config",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:llvm_compiler",
        "//xla/service/cpu:cpu_compiler_pure",
        "//xla/service/cpu:cpu_options",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)
