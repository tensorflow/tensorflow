# Fuzzing TensorFlow ops..
# Most ops have a similar set of dependencies and a similar fuzzing
# infrastructure. Hence, we gather everything in one single place.
# Note that these fuzzers cover a large part of TF, they are not granular.

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_fuzz_target",
)

package(
    licenses = ["notice"],
)

# Since all ops need to have a graph created before being fuzzed, we define
# this header-only library to handle the needed plumbing.
cc_library(
    name = "fuzz_session",
    hdrs = ["fuzz_session.h"],
    deps = [
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:status",
    ],
)

# A trivial fuzzer with no pre-specified corpus.
tf_fuzz_target(
    name = "identity_fuzz",
    srcs = ["identity_fuzz.cc"],
    tags = ["no_oss"],  # b/175698644
    deps = [
        ":fuzz_session",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:array",
    ],
)
