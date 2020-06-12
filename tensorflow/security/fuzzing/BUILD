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

# A trivial fuzzer with no pre-specified corpus.
# TODO(mihaimaruseac): Move fuzz_session and the op fuzzers to a subdirectory
tf_fuzz_target(
    name = "identity_fuzz",
    srcs = ["identity_fuzz.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels/fuzzing:fuzz_session",
        # Needed only to transitiviely link dependencies
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
    ],
)
