package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "tf_graph_controls",
    srcs = [
        "tf-graph-controls.html",
    ],
    path = "/tf-graph-controls",
    deps = [
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_graph_common",
        "@org_polymer",
        "@org_polymer_paper_button",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_menu",
        "@org_polymer_paper_radio_group",
        "@org_polymer_paper_toggle_button",
        "@org_polymer_paper_tooltip",
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
        "tf-graph-controls.html",
    ],
    destdir = "tf-graph-controls",
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
