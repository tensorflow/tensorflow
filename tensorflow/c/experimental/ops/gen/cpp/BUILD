load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "cpp",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(["*.h"]),
    visibility = ["//tensorflow/c/experimental/ops/gen:__pkg__"],
    deps = [
        "//tensorflow/c/experimental/ops/gen/common",
        "//tensorflow/c/experimental/ops/gen/cpp/renderers",
        "//tensorflow/c/experimental/ops/gen/cpp/views",
        "//tensorflow/c/experimental/ops/gen/model",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "cpp_generator_test",
    size = "small",
    srcs = ["cpp_generator_test.cc"],
    data = ["//tensorflow/c/experimental/ops/gen/cpp/golden"],
    deps = [
        ":cpp",
        "//tensorflow/c/experimental/ops/gen/common",
        "//tensorflow/c/experimental/ops/gen/cpp/renderers",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@local_tsl//tsl/platform:status",
    ],
)
