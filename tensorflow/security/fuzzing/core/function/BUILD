# Fuzzing TensorFlow/core/function with GFT

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_test",
)

tf_cc_shared_test(
    name = "runtime_client_fuzz",
    srcs = ["runtime_client_fuzz.cc"],
    deps = [
        "//tensorflow/core/function:runtime_client_cc",
        "//tensorflow/core/function/testing:test_pass_cc",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:fuzztest_gtest_main",
        "@com_google_googletest//:gtest_main",
    ],
)