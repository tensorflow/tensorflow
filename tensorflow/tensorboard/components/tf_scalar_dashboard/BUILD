package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "tf_scalar_dashboard",
    srcs = [
        "tf-scalar-dashboard.html",
        "tf-smoothing-input.html",
    ],
    path = "/tf-scalar-dashboard",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_color_scale",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/vz_line_chart",
        "@org_polymer",
        "@org_polymer_iron_collapse",
        "@org_polymer_paper_checkbox",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_input",
        "@org_polymer_paper_item",
        "@org_polymer_paper_menu",
        "@org_polymer_paper_slider",
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
        "tf-scalar-dashboard.html",
        "tf-smoothing-input.html",
    ],
    destdir = "tf-scalar-dashboard",
    deps = [
        "//tensorflow/tensorboard/components:tf_imports",
        "//tensorflow/tensorboard/components/tf_backend:legacy",
        "//tensorflow/tensorboard/components/tf_color_scale:legacy",
        "//tensorflow/tensorboard/components/tf_dashboard_common:legacy",
        "//tensorflow/tensorboard/components/vz_line_chart:legacy",
        "//third_party/javascript/polymer/v1/iron-collapse:lib",
        "//third_party/javascript/polymer/v1/paper-checkbox:lib",
        "//third_party/javascript/polymer/v1/paper-dropdown-menu:lib",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
        "//third_party/javascript/polymer/v1/paper-input:lib",
        "//third_party/javascript/polymer/v1/paper-item:lib",
        "//third_party/javascript/polymer/v1/paper-menu:lib",
        "//third_party/javascript/polymer/v1/paper-slider:lib",
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
