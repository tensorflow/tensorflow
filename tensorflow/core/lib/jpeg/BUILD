# Description:
# JPEG handle and mem packages.

load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "if_mobile",
    "tf_cc_test",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core:__pkg__"],
    features = ["-parse_headers"],
    licenses = ["notice"],
)

cc_library(
    name = "jpeg_internal",
    srcs = [
        "jpeg_handle.cc",
        "jpeg_mem.cc",
        "//tensorflow/core/platform:jpeg_hdrs",
    ],
    hdrs = [
        "jpeg_handle.h",
        "jpeg_mem.h",
    ],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:jpeg",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@libjpeg_turbo//:jpeg",
    ],
)

cc_library(
    name = "portable_jpeg_internal",
    srcs = if_mobile([
        "jpeg_handle.cc",
        "jpeg_mem.cc",
        "//tensorflow/core/platform:jpeg_hdrs",
    ]),
    hdrs = [
        "jpeg_handle.h",
        "jpeg_mem.h",
        "//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header",
        "//tensorflow/core/platform:jpeg_internal_hdrs",
        "@local_tsl//tsl/platform:jpeg_internal_hdrs",
        "@local_xla//xla/tsl/platform/default:integral_types.h",
    ],
    copts = tf_copts(),
    linkopts = if_android(["-ldl"]),
    deps = [
        "//tensorflow/core:core_stringpiece",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:float8",
        "//tensorflow/core/platform:jpeg",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@libjpeg_turbo//:jpeg",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:tstring",
        "@local_tsl//tsl/platform:types",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tf_cc_test(
    name = "lib_jpeg_jpeg_mem_unittest",
    srcs = ["jpeg_mem_unittest.cc"],
    data = ["//tensorflow/core/lib/jpeg/testdata"],
    deps = [
        ":jpeg_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:jpeg",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:check",
        "@libjpeg_turbo//:jpeg",
    ],
)
