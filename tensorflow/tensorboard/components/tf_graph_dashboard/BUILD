package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_graph_dashboard",
    srcs = [
        "tf-graph-dashboard.html",
    ],
    path = "/tf-graph-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_graph",
        "//tensorflow/tensorboard/components/tf_graph_board",
        "//tensorflow/tensorboard/components/tf_graph_controls",
        "//tensorflow/tensorboard/components/tf_graph_loader",
        "@org_polymer",
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
        "tf-graph-dashboard.html",
    ],
    destdir = "tf-graph-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_dashboard_common:legacy",
        "//tensorflow/tensorboard/components/tf_graph:legacy",
        "//tensorflow/tensorboard/components/tf_graph_board:legacy",
        "//tensorflow/tensorboard/components/tf_graph_controls:legacy",
        "//tensorflow/tensorboard/components/tf_graph_loader:legacy",
    ],
)

# This is needed despite how this component lacks TypeScript files because
# components/BUILD seeks a legacy_ts rule in this package.
tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [],
    deps_mgmt = "off",
    runtime = "nodejs",
)
