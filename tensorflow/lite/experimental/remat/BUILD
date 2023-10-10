load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/lite/...",
    ],
)

cc_library(
    name = "metadata_util",
    srcs = ["metadata_util.cc"],
    hdrs = ["metadata_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:graph_info",
    ],
)

cc_test(
    name = "metadata_util_test",
    size = "small",
    srcs = ["metadata_util_test.cc"],
    deps = [
        ":metadata_util",
        "@com_google_googletest//:gtest_main",
    ],
)
