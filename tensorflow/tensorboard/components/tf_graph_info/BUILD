package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_graph_info",
    srcs = [
        "tf-graph-icon.html",
        "tf-graph-info.html",
        "tf-node-info.html",
        "tf-node-list-item.html",
    ],
    path = "/tf-graph-info",
    deps = [
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_graph_common",
        "@org_polymer",
        "@org_polymer_iron_collapse",
        "@org_polymer_iron_list",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_item",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_spinner",
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
        "tf-graph-icon.html",
        "tf-graph-info.html",
        "tf-node-info.html",
        "tf-node-list-item.html",
    ],
    destdir = "tf-graph-info",
    deps = [
        "//tensorflow/tensorboard/components/tf_dashboard_common:legacy",
        "//tensorflow/tensorboard/components/tf_graph_common:legacy",
    ],
)

# This is needed despite how this component lacks TypeScript files because
# components/BUILD seeks a legacy_ts rule in this package.
tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [],
)
