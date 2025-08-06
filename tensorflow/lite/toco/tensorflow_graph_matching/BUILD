load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "cluster_utils",
    srcs = [
        "cluster_utils.cc",
    ],
    hdrs = [
        "cluster_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "cluster",
    srcs = [
        "cluster.cc",
    ],
    hdrs = [
        "cluster.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":cluster_utils",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
    ],
)

cc_library(
    name = "resolve_svdf",
    srcs = [
        "resolve_svdf.cc",
    ],
    hdrs = [
        "resolve_svdf.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:toco_port",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

tf_cc_test(
    name = "resolve_svdf_test",
    srcs = ["resolve_svdf_test.cc"],
    deps = [
        ":cluster",
        ":cluster_utils",
        ":resolve_cluster",
        ":resolve_svdf",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:toco_port",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "resolve_cluster",
    srcs = [
        "resolve_cluster.cc",
    ],
    hdrs = [
        "resolve_cluster.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_utils",
        ":resolve_svdf",
        "//tensorflow/core:protos_all_cc",
    ],
)
