load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "views",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(["*.h"]),
    visibility = [
        "//tensorflow/c/experimental/ops/gen/cpp:__pkg__",
        "//tensorflow/c/experimental/ops/gen/cpp/renderers:__pkg__",
    ],
    deps = [
        "//tensorflow/c/experimental/ops/gen/common",
        "//tensorflow/c/experimental/ops/gen/model",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
