load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "generator",
    srcs = ["generator.cc"],
    hdrs = ["generator.h"],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "generate",
    srcs = ["generate.cc"],
    deps = [
        ":generator",
    ],
)

cc_test(
    name = "generator_test",
    srcs = ["generator_test.cc"],
    deps = [
        ":generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "consistency_test",
    srcs = ["consistency_test.cc"],
    data = [
        "//tensorflow/lite:builtin_ops.h",
    ],
    deps = [
        ":generator",
        "@com_google_googletest//:gtest_main",
    ],
)
