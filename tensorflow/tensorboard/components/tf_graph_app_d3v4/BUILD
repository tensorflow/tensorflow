package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_graph_app_d3v4",
    srcs = [
        "index.html",
        "tf-graph-app.html",
    ],
    path = "/tf-graph-app",
    deps = [
        "//tensorflow/tensorboard/components/tf_graph_board_d3v4",
        "//tensorflow/tensorboard/components/tf_graph_controls_d3v4",
        "//tensorflow/tensorboard/components/tf_graph_loader_d3v4",
        "@org_polymer",
        "@org_polymer_iron_component_page",
        "@org_polymer_webcomponentsjs",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
