package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "tf_image_dashboard",
    srcs = [
        "tf-image-dashboard.html",
        "tf-image-loader.html",
    ],
    path = "/tf-image-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_color_scale",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "@org_polymer",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_icon_button",
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
        "tf-image-dashboard.html",
        "tf-image-loader.html",
    ],
    destdir = "tf-image-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend:legacy",
        "//tensorflow/tensorboard/components/tf_dashboard_common:legacy",
    ],
)

# This is needed: components/BUILD seeks a legacy_ts rule in this package.
tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [],
)
