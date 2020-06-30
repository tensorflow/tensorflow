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
    name = "status_fuzz",
    srcs = ["status_fuzz.cc"],
    deps = [
        "//tensorflow/core/platform:status",
    ],
)

tf_fuzz_target(
    name = "status_group_fuzz",
    srcs = ["status_group_fuzz.cc"],
    deps = [
        "//tensorflow/core/platform:status",
    ],
)
