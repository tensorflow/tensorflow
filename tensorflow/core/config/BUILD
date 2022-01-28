load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [
        "//tensorflow/core/config:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "flag_defs",
    srcs = ["flag_defs.cc"],
    hdrs = ["flag_defs.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":flags",
    ],
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/util:env_var",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "flags_test",
    srcs = ["flags_test.cc"],
    deps = [
        ":flag_defs",
        ":flags",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
