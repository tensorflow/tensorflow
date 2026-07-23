load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_test(
    name = "pad_fuzz_test",
    srcs = ["pad_fuzz_test.cc"],
    args = ["--heap_check="],
    copts = ["-std=c++20"],
    tags = ["no_heap_check"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_fuzztest//fuzztest",
        "@com_google_googletest//:gtest_main_no_heapcheck",
        "@flatbuffers",
    ],
)
