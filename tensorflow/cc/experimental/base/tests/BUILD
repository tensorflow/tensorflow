load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Tests for the C++ header-only base types.
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    licenses = ["notice"],
)

cc_library(
    name = "tensor_types_test_util",
    testonly = True,
    hdrs = ["tensor_types_test_util.h"],
    deps = [
        "//tensorflow/c:tf_datatype",
    ],
)

tf_cc_test(
    name = "tensor_test",
    srcs = [
        "tensor_test.cc",
    ],
    deps = [
        ":tensor_types_test_util",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/cc/experimental/base/public:status",
        "//tensorflow/cc/experimental/base/public:tensor",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "tensorhandle_test",
    srcs = [
        "tensorhandle_test.cc",
    ],
    deps = [
        ":tensor_types_test_util",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/cc/experimental/base/public:runtime",
        "//tensorflow/cc/experimental/base/public:runtime_builder",
        "//tensorflow/cc/experimental/base/public:status",
        "//tensorflow/cc/experimental/base/public:tensor",
        "//tensorflow/cc/experimental/base/public:tensorhandle",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
