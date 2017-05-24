package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_graph_board",
    srcs = [
        "tf-graph-board.html",
    ],
    path = "/tf-graph-board",
    deps = [
        "//tensorflow/tensorboard/components/tf_graph",
        "//tensorflow/tensorboard/components/tf_graph_common",
        "//tensorflow/tensorboard/components/tf_graph_info",
        "@org_polymer",
        "@org_polymer_paper_progress",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)

################################################################################
# MARKED FOR DELETION

load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [
        "tf-graph-board.html",
    ],
    destdir = "tf-graph-board",
    deps = [
        "//tensorflow/tensorboard/components/tf_graph:legacy",
        "//tensorflow/tensorboard/components/tf_graph_common:legacy",
        "//tensorflow/tensorboard/components/tf_graph_info:legacy",
    ],
)
