load("//xla:package_groups.bzl", "xla_test_friend_package_group")
load("//xla:restricted_package.bzl", "xla_restricted_allow", "xla_restricted_verify")
load("//xla/tests:build_defs.bzl", "generate_backend_suites", "xla_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

xla_test_friend_package_group(name = "friends")

generate_backend_suites()

cc_library(
    name = "hlo_test_base_legacy",
    testonly = True,
    srcs = ["hlo_test_base_legacy.cc"],
    hdrs = ["hlo_test_base_legacy.h"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/service:backend",
        "//xla/service:compiler",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_runner_interface",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/service:platform_util",
        "//xla/service/restricted:hlo_runner_legacy",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_address_allocator",
        "//xla/tests:hlo_runner_agnostic_reference_mixin",
        "//xla/tests:hlo_runner_agnostic_test_base",
        "//xla/tests:pjrt_client_registry",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "llvm_compiler_test",
    srcs = ["llvm_compiler_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    use_legacy_runtime = True,
    deps = [
        ":hlo_test_base_legacy",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:backend",
        "//xla/service:llvm_compiler",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "llvm_irgen_test_base",
    testonly = True,
    srcs = ["llvm_irgen_test_base.cc"],
    hdrs = ["llvm_irgen_test_base.h"],
    deps = [
        ":hlo_test_base_legacy",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/service:executable",
        "//xla/service:llvm_compiler",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_for_library",
        "@tsl//tsl/platform:test",
    ],
)

xla_restricted_verify(allowed_targets = [
    xla_restricted_allow("hlo_test_base_legacy"),
    xla_restricted_allow("llvm_compiler_test"),
    xla_restricted_allow("llvm_irgen_test_base"),
])
