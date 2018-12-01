package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

cc_library(
    name = "cluster_utils",
    srcs = [
        "cluster_utils.cc",
    ],
    hdrs = [
        "cluster_utils.h",
    ],
    deps = [
        "//tensorflow/lite/toco:toco_port",
    ],
)

cc_library(
    name = "cluster",
    srcs = [
        "cluster.cc",
    ],
    hdrs = [
        "cluster.h",
    ],
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
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:toco_port",
        "//tensorflow/lite/toco:tooling_util",
        "@protobuf_archive//:protobuf_headers",
    ],
)

tf_cc_test(
    name = "resolve_svdf_test",
    srcs = ["resolve_svdf_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":cluster",
        ":cluster_utils",
        ":resolve_cluster",
        ":resolve_svdf",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
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
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_utils",
        ":resolve_svdf",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:tooling_util",
    ],
)
