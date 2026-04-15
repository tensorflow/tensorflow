# Raw-bytes fuzzer for the TensorFlow V2 checkpoint loader.

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_fuzz_test(
    name = "load_checkpoint_fuzz",
    srcs = ["load_checkpoint_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/c:tf_status",
    ],
)
