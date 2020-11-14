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

tf_fuzz_target(
    name = "consume_leading_digits_fuzz",
    srcs = ["consume_leading_digits_fuzz.cc"],
    deps = [
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringpiece",
    ],
)

tf_fuzz_target(
    name = "arg_def_case_fuzz",
    srcs = ["arg_def_case_fuzz.cc"],
    deps = [
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringpiece",
    ],
)

tf_fuzz_target(
    name = "string_replace_fuzz",
    srcs = ["string_replace_fuzz.cc"],
    deps = [
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringpiece",
    ],
)

tf_fuzz_target(
    name = "stringprintf_fuzz",
    srcs = ["stringprintf_fuzz.cc"],
    deps = [
        "//tensorflow/core/platform:stringprintf",
    ],
)

tf_fuzz_target(
    name = "tstring_fuzz",
    srcs = ["tstring_fuzz.cc"],
    deps = [
        "//tensorflow/core/platform:tstring",
    ],
)

tf_fuzz_target(
    name = "base64_fuzz",
    srcs = ["base64_fuzz.cc"],
    deps = [
        "//tensorflow/core/platform:base64",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringpiece",
    ],
)
