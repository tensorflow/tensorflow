load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:internal",
        "//tensorflow/compiler/xla:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "graphcycles",
    srcs = ["graphcycles.cc"],
    hdrs = ["graphcycles.h"],
    deps = [
        ":ordered_set",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ordered_set",
    hdrs = ["ordered_set.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "ordered_set_test",
    srcs = ["ordered_set_test.cc"],
    deps = [
        ":ordered_set",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
