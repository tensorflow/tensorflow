load("//xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "sharding_conversions",
    srcs = ["sharding_conversions.cc"],
    hdrs = ["sharding_conversions.h"],
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/python/ifrt",
        "//xla/python/ifrt/ir:sharding_param",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "sharding_conversions_test",
    srcs = ["sharding_conversions_test.cc"],
    deps = [
        ":sharding_conversions",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/python/ifrt",
        "//xla/python/ifrt:mock",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/ir:sharding_param",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)
