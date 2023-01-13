# Fuzzing TensorFlow/core/framework with GFT

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

tf_cc_fuzz_test(
    name = "function_fuzz",
    srcs = ["function_fuzz.cc"],
    deps = [
        "//tensorflow/core/framework:function_fuzz",
        "//tensorflow/security/fuzzing/domains:function_def",
    ],
)