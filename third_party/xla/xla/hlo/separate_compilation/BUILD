# Targets related to hlo_module_splitting, linking, and more generally
# incremental compilation of HLO.

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "hlo_linking_manifest",
    hdrs = ["hlo_linking_manifest.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:compilation_environments",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "hlo_module_splitting",
    srcs = ["hlo_module_splitting.cc"],
    hdrs = ["hlo_module_splitting.h"],
    deps = [
        ":hlo_linking_manifest",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/service:compilation_environments",
        "//xla/service:hlo_module_config",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hlo_module_linking",
    srcs = ["hlo_module_linking.cc"],
    hdrs = ["hlo_module_linking.h"],
    deps = [
        ":hlo_linking_manifest",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/service:compilation_environments",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# Tests below

xla_cc_test(
    name = "hlo_module_splitting_test",
    size = "small",
    srcs = ["hlo_module_splitting_test.cc"],
    deps = [
        ":hlo_module_splitting",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:compiler",
        "//xla/service:cpu_plugin",  # fixdeps: keep
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "hlo_module_linking_test",
    size = "small",
    srcs = ["hlo_module_linking_test.cc"],
    deps = [
        ":hlo_linking_manifest",
        ":hlo_module_linking",
        ":hlo_module_splitting",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:compiler",
        "//xla/service:cpu_plugin",  # fixdeps: keep
        "//xla/service:hlo_verifier",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
