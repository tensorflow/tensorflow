# Fuzzing TensorFlow ops with GFT
# Most ops have a similar set of dependencies and a similar fuzzing
# infrastructure. Hence, we gather everything in one single place.
# Note that these fuzzers cover a large part of TF, they are not granular.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

# Since all ops need to have a graph created before being fuzzed, we define
# this header-only library to handle the needed plumbing.
cc_library(
    name = "fuzz_session",
    testonly = 1,
    hdrs = ["fuzz_session.h"],
    tags = ["no_oss"],
    visibility = ["//tensorflow/cc/framework/fuzzing:__subpackages__"],
    deps = [
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:status",
        "@com_google_fuzztest//fuzztest",
    ],
)

# A trivial fuzzer with no pre-specified corpus.
tf_cc_test(
    name = "identity_fuzz",
    srcs = ["identity_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        ":fuzz_session",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:array",
        "@com_google_fuzztest//fuzztest",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "concat_fuzz",
    srcs = ["concat_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        ":fuzz_session",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:array",
        "@com_google_fuzztest//fuzztest",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "add_fuzz",
    srcs = ["add_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        ":fuzz_session",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:array",
        "@com_google_fuzztest//fuzztest",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "matmul_fuzz",
    srcs = ["matmul_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        ":fuzz_session",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:matmul_op",
        "@com_google_fuzztest//fuzztest",
        "@com_google_googletest//:gtest_main",
    ],
)
