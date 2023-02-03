# Fuzzing TensorFlow/core/function with GFT

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

tf_cc_fuzz_test(
    name = "runtime_client_fuzz",
    srcs = ["runtime_client_fuzz.cc"],
    deps = [
        "//tensorflow/core/function/runtime_client:runtime_client_cc",
    ],
)
