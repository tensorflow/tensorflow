package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_graph_app",
    srcs = [
        "index.html",
        "tf-graph-app.html",
    ],
    path = "/tf-graph-app",
    deps = [
        "//tensorflow/tensorboard/components/tf_graph_board",
        "//tensorflow/tensorboard/components/tf_graph_controls",
        "//tensorflow/tensorboard/components/tf_graph_loader",
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

################################################################################
# MARKED FOR DELETION

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [
        "index.html",
        "tf-graph-app.html",
    ],
    destdir = "tf-graph-app",
    deps = [
        "//tensorflow/tensorboard/components/tf_graph_board:legacy",
        "//tensorflow/tensorboard/components/tf_graph_controls:legacy",
        "//tensorflow/tensorboard/components/tf_graph_loader:legacy",
        "//third_party/javascript/polymer/v1/iron-list:lib",
        "//third_party/javascript/polymer/v1/paper-radio-group:lib",
        "//third_party/javascript/polymer/v1/paper-tooltip:lib",
    ],
)
