load("@bazel_skylib//lib:selects.bzl", "selects")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/compiler/mlir/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "compatibility_macros",
    hdrs = ["compatibility_macros.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_library(
    name = "runtime_shape",
    srcs = ["runtime_shape.cc"],
    hdrs = ["runtime_shape.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [":compatibility_macros"],
)

tf_cc_test(
    name = "runtime_shape_test",
    srcs = ["runtime_shape_test.cc"],
    deps = [
        ":runtime_shape",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

# LINT.IfChange

cc_library(
    name = "cppmath",
    srcs = [],
    hdrs = [
        "cppmath.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":compatibility_macros",
        ":cppmath",
    ],
)

config_setting(
    name = "haswell",
    values = {
        "cpu": "haswell",
    },
)

config_setting(
    name = "ios_x86_64",
    values = {
        "cpu": "ios_x86_64",
    },
)

config_setting(
    name = "tvos_x86_64",
    values = {
        "cpu": "tvos_x86_64",
    },
)

config_setting(
    name = "k8",
    values = {
        "cpu": "k8",
    },
)

config_setting(
    name = "x86",
    values = {
        "cpu": "x86",
    },
)

config_setting(
    name = "x86_64",
    values = {
        "cpu": "x86_64",
    },
)

config_setting(
    name = "darwin",
    values = {
        "cpu": "darwin",
    },
)

config_setting(
    name = "darwin_x86_64",
    values = {
        "cpu": "darwin_x86_64",
    },
)

config_setting(
    name = "freebsd",
    values = {
        "cpu": "freebsd",
    },
)

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)

selects.config_setting_group(
    name = "x86_any",
    match_any = [
        ":haswell",
        ":ios_x86_64",
        ":k8",
        ":x86",
        ":x86_64",
        ":darwin",
        ":darwin_x86_64",
        ":freebsd",
        ":windows",
        ":tvos_x86_64",
    ],
)

cc_library(
    name = "cpu_check",
    hdrs = [
        "optimized/neon_check.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        ":x86_any": ["@arm_neon_2_x86_sse"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":compatibility_macros",
        ":cpu_check",
        "//tensorflow/compiler/mlir/lite/core:macros",
        "@gemmlowp//:fixedpoint",
    ],
)

# LINT.ThenChange(//tensorflow/lite/kernels/internal/BUILD)
