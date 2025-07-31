load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_test(
    name = "gpu_delegate_compatibility_checker_test",
    srcs = [
        "gpu_delegate_compatibility_checker_test.cc",
    ],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/conv3d_huge_im2col.bin",
    ],
    tags = ["no_oss"],  #TODO(b/276295784): Re-enable when fixed.
    deps = [
        ":gpu_delegate_compatibility_checker",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_delegate_compatibility_checker",
    srcs = ["gpu_delegate_compatibility_checker.cc"],
    hdrs = [
        "gpu_delegate_compatibility_checker.h",
    ],
    deps = [
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/tools/delegates/compatibility/common:delegate_compatibility_checker_base",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tensorflow/lite/tools/versioning:gpu_compatibility",
        "//tensorflow/lite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
    ],
)
