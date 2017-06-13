package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "tf_graph_board",
    srcs = ["tf-graph-board.html"],
    path = "/tf-graph-board",
    deps = [
        "//tensorflow/tensorboard/components/tf_graph",
        "//tensorflow/tensorboard/components/tf_graph_common",
        "//tensorflow/tensorboard/components/tf_graph_info",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "@org_polymer_paper_progress",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_graph_board"],
    destdir = "tf-graph-board",
    deps = [
        "//tensorflow/tensorboard/components/tf_graph:legacy",
        "//tensorflow/tensorboard/components/tf_graph_common:legacy",
        "//tensorflow/tensorboard/components/tf_graph_info:legacy",
        "//third_party/javascript/polymer/v1/paper-progress:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
