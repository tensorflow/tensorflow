load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "python_utils",
    srcs = ["python_utils.cc"],
    hdrs = ["python_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "python_error_reporter",
    srcs = ["python_error_reporter.cc"],
    hdrs = ["python_error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/lite:stateful_error_reporter",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)
