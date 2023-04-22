# Description:
# gif io package.

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
    default_visibility = ["//tensorflow/core:__pkg__"],
    features = ["-parse_headers"],
    licenses = ["notice"],
)

cc_library(
    name = "gif_internal",
    srcs = [
        "gif_io.cc",
        "//tensorflow/core/platform:gif_hdrs",
    ],
    hdrs = ["gif_io.h"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:freebsd": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:gif",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "portable_gif_internal",
    srcs = if_mobile([
        "gif_io.cc",
        "//tensorflow/core/platform:gif_hdrs",
    ]),
    hdrs = [
        "gif_io.h",
        "//tensorflow/core/lib/core:legacy_lib_core_stringpiece_header",
        "//tensorflow/core/lib/gtl:legacy_android_gif_internal_headers",
        "//tensorflow/core/platform:gif_internal_hdrs",
        "//tensorflow/core/platform/default:integral_types.h",
        "//tensorflow/core/platform/default:logging.h",
    ],
    copts = tf_copts(),
    linkopts = if_android(["-ldl"]),
    deps = [
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:gif",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "lib_gif_io_test",
    srcs = ["gif_io_test.cc"],
    data = ["//tensorflow/core/lib/gif/testdata:gif_testdata"],
    deps = [
        ":gif_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/png:png_io",
        "@com_google_absl//absl/base",
    ],
)
