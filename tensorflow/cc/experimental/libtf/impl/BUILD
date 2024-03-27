# libtf implementation details.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/cc/experimental/libtf:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "iostream",
    srcs = [
        "iostream.cc",
    ],
    deps = [
        ":none",
        ":string",
        ":tensor_spec",
    ],
)

tf_cc_test(
    name = "iostream_test",
    size = "small",
    srcs = ["iostream_test.cc"],
    deps = [
        ":iostream",
        ":none",
        ":scalars",
        ":string",
        ":tensor_spec",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "scalars",
    hdrs = [
        "scalars.h",
    ],
)

tf_cc_test(
    name = "scalars_test",
    size = "small",
    srcs = ["scalars_test.cc"],
    deps = [
        ":scalars",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "string",
    srcs = [
        "string.cc",
    ],
    hdrs = [
        "string.h",
    ],
)

tf_cc_test(
    name = "string_test",
    size = "small",
    srcs = ["string_test.cc"],
    deps = [
        ":string",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "none",
    srcs = [
        "none.cc",
    ],
    hdrs = [
        "none.h",
    ],
)

tf_cc_test(
    name = "none_test",
    size = "small",
    srcs = ["none_test.cc"],
    deps = [
        ":none",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "tensor_spec",
    hdrs = [
        "tensor_spec.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "tensor_spec_test",
    size = "small",
    srcs = ["tensor_spec_test.cc"],
    deps = [
        ":iostream",  # Necessary for absl::VerifyTypeImplementsAbslHashCorrectly.
        ":tensor_spec",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/hash:hash_testing",
    ],
)
