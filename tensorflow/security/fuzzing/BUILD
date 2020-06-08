# Fuzzing TensorFlow.
# Since we use OSSFuzz, gather all fuzzers into a single place.
# This way, we can use tooling to determine the status of the fuzzing efforts.

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_fuzz_target",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_fuzz_target(
    name = "demo_fuzz",
    srcs = ["demo_fuzz.cc"],
)
