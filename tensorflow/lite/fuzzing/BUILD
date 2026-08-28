# Fuzzing harnesses for the TensorFlow Lite runtime.

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

tf_cc_fuzz_test(
    name = "interpreter_fuzz",
    srcs = ["interpreter_fuzz.cc"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)
