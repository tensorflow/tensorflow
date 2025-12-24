load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/backends/gpu/target_config:build_defs.bzl", "embed_files")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_gpu_specs",
    data = glob(["specs/*.txtpb"]),
)

exports_files(glob([
    "specs/*.txtpb",
]))

embed_files(
    name = "embed_gpu_specs",
    srcs = glob(["specs/*.txtpb"]),
    cpp_namespace = "xla::gpu",
)

cc_library(
    name = "target_config",
    srcs = ["target_config.cc"],
    hdrs = ["target_config.h"],
    deps = [
        ":embed_gpu_specs",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

xla_cc_test(
    name = "target_config_test",
    srcs = ["target_config_test.cc"],
    deps = [
        ":target_config",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla/tsl:package_groups_bzl",
        "//xla/tsl:tsl_default_bzl",
        "//xla/tsl/platform:rules_cc_bzl",
    ],
)
