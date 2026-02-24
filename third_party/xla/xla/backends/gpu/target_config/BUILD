load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/util:build_defs.bzl", "embed_files")

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
    compatible_with = get_compatible_with_portable() + if_google(["//buildenv/target:libtpu"]),
    cpp_namespace = "xla::gpu",
)

cc_library(
    name = "target_config",
    srcs = ["target_config.cc"],
    hdrs = ["target_config.h"],
    compatible_with = get_compatible_with_portable() + if_google(["//buildenv/target:libtpu"]),
    deps = [
        ":embed_gpu_specs",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream_executor_h",
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
        "@com_google_protobuf//:protobuf",
    ],
)
