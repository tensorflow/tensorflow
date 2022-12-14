# Fuzzing TensorFlow/core/function with GFT

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_test",
)

tf_cc_shared_test(
    name = "runtime_client_fuzz",
    srcs = ["runtime_client_fuzz.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@com_google_fuzztest//fuzztest",
        "//tensorflow/core/function:runtime_client_cc",
#        "//tensorflow/core/kernels:array",
    ],
)