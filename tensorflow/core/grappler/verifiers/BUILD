load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "graph_verifier.h",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
)

cc_library(
    name = "graph_verifier",
    hdrs = [
        "graph_verifier.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "structure_verifier",
    srcs = ["structure_verifier.cc"],
    hdrs = [
        "structure_verifier.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_verifier",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:topological_sort",
    ],
)

tf_cc_test(
    name = "structure_verifier_test",
    srcs = ["structure_verifier_test.cc"],
    deps = [
        ":structure_verifier",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "@com_google_absl//absl/strings",
    ],
)
