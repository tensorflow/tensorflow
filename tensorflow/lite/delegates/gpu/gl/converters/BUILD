load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bhwc_to_phwc4",
    srcs = ["bhwc_to_phwc4.cc"],
    hdrs = ["bhwc_to_phwc4.h"],
    deps = [
        ":util",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:command_queue",
        "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "//tensorflow/lite/delegates/gpu/gl:gl_program",
        "//tensorflow/lite/delegates/gpu/gl:gl_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
    ],
)

cc_test(
    name = "bhwc_to_phwc4_test",
    size = "small",
    srcs = ["bhwc_to_phwc4_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv2",
    ],
    tags = [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":bhwc_to_phwc4",
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "//tensorflow/lite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "phwc4_to_bhwc",
    srcs = ["phwc4_to_bhwc.cc"],
    hdrs = ["phwc4_to_bhwc.h"],
    deps = [
        ":util",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:command_queue",
        "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "//tensorflow/lite/delegates/gpu/gl:gl_program",
        "//tensorflow/lite/delegates/gpu/gl:gl_shader",
        "//tensorflow/lite/delegates/gpu/gl:variable",
    ],
)

cc_test(
    name = "phwc4_to_bhwc_test",
    size = "small",
    srcs = ["phwc4_to_bhwc_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv2",
    ],
    tags = [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":phwc4_to_bhwc",
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "//tensorflow/lite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
