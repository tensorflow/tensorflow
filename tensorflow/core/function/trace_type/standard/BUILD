load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    licenses = ["notice"],
)

cc_library(
    name = "trace_type",
    hdrs = ["trace_type.h"],
)

cc_library(
    name = "primitive_types",
    srcs = ["primitive_types.cc"],
    hdrs = ["primitive_types.h"],
    deps = [
        ":trace_type",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "primitive_types_test",
    srcs = ["primitive_types_test.cc"],
    deps = [
        ":primitive_types",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
