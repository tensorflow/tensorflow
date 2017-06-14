package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "tf_graph_dashboard",
    srcs = ["tf-graph-dashboard.html"],
    path = "/tf-graph-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_graph",
        "//tensorflow/tensorboard/components/tf_graph_board",
        "//tensorflow/tensorboard/components/tf_graph_controls",
        "//tensorflow/tensorboard/components/tf_graph_loader",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "//tensorflow/tensorboard/components/vz_sorting",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_graph_dashboard"],
    destdir = "tf-graph-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend:legacy",
        "//tensorflow/tensorboard/components/tf_dashboard_common:legacy",
        "//tensorflow/tensorboard/components/tf_graph:legacy",
        "//tensorflow/tensorboard/components/tf_graph_board:legacy",
        "//tensorflow/tensorboard/components/tf_graph_controls:legacy",
        "//tensorflow/tensorboard/components/tf_graph_loader:legacy",
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
