load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "connected_components",
    srcs = ["connected_components.cc"],
    hdrs = ["connected_components.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "connected_components_test",
    srcs = ["connected_components_test.cc"],
    deps = [
        ":connected_components",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_diff_util",
    hdrs = ["hlo_diff_util.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:fingerprint",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/tools/hlo_diff:hlo_gumgraph_mappings",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph",
        "//xla/hlo/tools/hlo_diff/graph:hlo_gumgraph_node",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_for_library",
    ],
)
