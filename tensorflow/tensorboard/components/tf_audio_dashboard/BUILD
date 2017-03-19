package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "tf_audio_dashboard",
    srcs = [
        "tf-audio-dashboard.html",
        "tf-audio-grid.html",
        "tf-audio-loader.html",
    ],
    path = "/tf-audio-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "@org_polymer",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_spinner",
        "@org_polymer_paper_styles",
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
        "tf-audio-dashboard.html",
        "tf-audio-grid.html",
        "tf-audio-loader.html",
    ],
    destdir = "tf-audio-dashboard",
    deps = [
        "//tensorflow/tensorboard/components:tf_imports",
        "//tensorflow/tensorboard/components/tf_backend:legacy",
        "//tensorflow/tensorboard/components/tf_dashboard_common:legacy",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
        "//third_party/javascript/polymer/v1/paper-styles:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [
    ],
    deps = ["//tensorflow/tensorboard/components:common_deps"],
)
