package(default_visibility = ["//tensorflow:internal"])

load(
    "//tensorflow/tensorboard:defs.bzl",
    "tensorboard_ts_development_sources",
    "tensorboard_ts_devserver",
    "tensorboard_ts_library",
    "tensorboard_webcomponent_library",
)

licenses(["notice"])  # Apache 2.0

tensorboard_webcomponent_library(
    name = "tf_dashboard_common",
    srcs = [
        "dashboard-style.html",
        "run-color-style.html",
        "scrollbar-style.html",
        "tensorboard-color.html",
        "tf-categorizer.html",
        "tf-collapsable-pane.html",
        "tf-dashboard.html",
        "tf-dashboard-layout.html",
        "tf-downloader.html",
        "tf-multi-checkbox.html",
        "tf-no-data-warning.html",
        "tf-option-selector.html",
        "tf-panes-helper.html",
        "tf-regex-group.html",
        "tf-run-selector.html",
        "tf-sidebar-helper.html",
    ],
    ts_lib_deps = [":ts"],
    deps = [
        "//third_party/javascript/plottable/v3:lib",
        "//third_party/javascript/polymer/v1/iron-ajax:lib",
        "//third_party/javascript/polymer/v1/iron-collapse:lib",
        "//third_party/javascript/polymer/v1/iron-icons:lib",
        "//third_party/javascript/polymer/v1/paper-button:lib",
        "//third_party/javascript/polymer/v1/paper-checkbox:lib",
        "//third_party/javascript/polymer/v1/paper-dialog:lib",
        "//third_party/javascript/polymer/v1/paper-dropdown-menu:lib",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
        "//third_party/javascript/polymer/v1/paper-input:lib",
        "//third_party/javascript/polymer/v1/paper-item:lib",
        "//third_party/javascript/polymer/v1/paper-menu:lib",
        "//third_party/javascript/polymer/v1/paper-slider:lib",
        "//third_party/javascript/polymer/v1/paper-spinner:lib",
        "//third_party/javascript/polymer/v1/paper-styles:lib",
        "//third_party/javascript/polymer/v1/paper-toggle-button:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "ts",
    srcs = [
        "dashboard-behavior.ts",
        "reload-behavior.ts",
        "tf-categorizer.ts",
        "tf-multi-checkbox.ts",
        "tf-regex-group.ts",
    ],
    deps = [
        "//tensorflow/tensorboard/components:common_deps_d3v4",
        "//tensorflow/tensorboard/components/tf_storage_d3v4:ts",
        "//tensorflow/tensorboard/components/vz_sorting_d3v4:ts",
    ],
)

tensorboard_ts_library(
    name = "tests",
    srcs = ["tf-categorizer-tests.ts"],
    deps = [
        ":ts",
        "//tensorflow/tensorboard/components:common_deps_d3v4",
    ],
)

tensorboard_webcomponent_library(
    name = "demo",
    srcs = [
        "tf-categorizer-demo.html",
        "tf-collapsable-pane-demo.html",
        "tf-multi-checkbox-demo.html",
        "tf-regex-group-demo.html",
    ],
    deps = [
        ":tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_color_scale_d3v4:tf_color_scale",
        "//third_party/javascript/polymer/v1/iron-demo-helpers:lib",
        "//third_party/javascript/polymer/v1/paper-styles:lib",
    ],
)

tensorboard_ts_devserver(
    name = "devserver",
    manifest = ":dev_sources",
    serving_path = "/demo_out/bundle.js",
    static_files = [":demo"],
)

tensorboard_ts_development_sources(
    name = "dev_sources",
    deps = [
        ":ts",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
