licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "graphcycles",
    srcs = ["graphcycles.cc"],
    hdrs = ["graphcycles.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "graphcycles_test",
    srcs = ["graphcycles_test.cc"],
    deps = [
        ":graphcycles",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
