load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/experimental/dtensor:__subpackages__",
        "//learning/brain/google/xla:__subpackages__",
        "//learning/brain/research/pjrt:__subpackages__",
        "//learning/brain/tfrc/executor:__subpackages__",
        "//learning/brain/tfrt/tpu_plugin:__subpackages__",
        "//tensorflow/compiler/jit:__subpackages__",
        "//tensorflow/compiler/mlir:__subpackages__",
        "//xla:__subpackages__",
        "//xla/backends/profiler/tpu:__subpackages__",
        "//tensorflow/core/common_runtime/next_pluggable_device:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/dtensor:__subpackages__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "tpu_compiler_variant",
    srcs = ["tpu_compiler_variant.cc"],
    hdrs = ["tpu_compiler_variant.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_compiler_variant",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "tpu_compiler_variant_test",
    srcs = ["tpu_compiler_variant_test.cc"],
    deps = [
        ":tpu_compiler_variant",
        "//xla/pjrt:pjrt_compiler_variant",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
