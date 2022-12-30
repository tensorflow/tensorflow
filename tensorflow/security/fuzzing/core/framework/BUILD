# Fuzzing TensorFlow/core/framework with GFT

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_test",
)

tf_cc_shared_test(
    name = "function_fuzz",
    srcs = ["function_fuzz.cc"],
    deps = [
        "//tensorflow/core/framework:function_fuzz",
        "//tensorflow/security/fuzzing/domains:function_def",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest_main",
    ],
)