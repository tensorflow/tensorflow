# Fuzzing TensorFlow ops with GFT
# Most ops have a similar set of dependencies and a similar fuzzing
# infrastructure. Hence, we gather everything in one single place.
# Note that these fuzzers cover a large part of TF, they are not granular.

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

# A trivial fuzzer with no pre-specified corpus.
tf_cc_fuzz_test(
    name = "identity_fuzz",
    srcs = ["identity_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:array",
        "//tensorflow/security/fuzzing/cc:fuzz_session",
    ],
)

tf_cc_fuzz_test(
    name = "concat_fuzz",
    srcs = ["concat_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:array",
        "//tensorflow/security/fuzzing/cc:fuzz_session",
    ],
)

tf_cc_fuzz_test(
    name = "add_fuzz",
    srcs = ["add_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:array",
        "//tensorflow/security/fuzzing/cc:fuzz_session",
        "//tensorflow/security/fuzzing/cc/core/framework:datatype_domains",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_domains",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_shape_domains",
    ],
)

tf_cc_fuzz_test(
    name = "matmul_fuzz",
    srcs = ["matmul_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/security/fuzzing/cc:fuzz_session",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_domains",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_shape_domains",
    ],
)

tf_cc_fuzz_test(
    name = "bincount_fuzz",
    srcs = ["bincount_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:bincount_op",
        "//tensorflow/security/fuzzing/cc:fuzz_session",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_domains",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_shape_domains",
    ],
)

tf_cc_fuzz_test(
    name = "string_to_number_fuzz",
    srcs = ["string_to_number_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:string_to_number_op",
        "//tensorflow/security/fuzzing/cc:fuzz_session",
    ],
)

tf_cc_fuzz_test(
    name = "string_ops_fuzz",
    srcs = ["string_ops_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:string",
        "//tensorflow/core/kernels:string_split_op",
        "//tensorflow/security/fuzzing/cc:fuzz_session",
    ],
)
