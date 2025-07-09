load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/platform/default:build_config.bzl", _strict_cc_test_impl = "strict_cc_test")

# re-expose for convenience
strict_cc_test = _strict_cc_test_impl

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

cc_library(
    name = "host_offloading_allocator",
    hdrs = ["host_offloading_allocator.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "host_offloading_buffer",
    hdrs = ["host_offloading_buffer.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "host_offloading_layout_analysis",
    srcs = ["host_offloading_layout_analysis.cc"],
    hdrs = ["host_offloading_layout_analysis.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:computation_layout",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

strict_cc_test(
    name = "host_offloading_layout_analysis_test",
    srcs = ["host_offloading_layout_analysis_test.cc"],
    deps = [
        ":host_offloading_layout_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "host_offloading_transforms",
    srcs = ["host_offloading_transforms.cc"],
    hdrs = ["host_offloading_transforms.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_layout",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

strict_cc_test(
    name = "host_offloading_transforms_test",
    srcs = ["host_offloading_transforms_test.cc"],
    deps = [
        ":host_offloading_transforms",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
