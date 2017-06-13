package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "tf_dashboard_common",
    srcs = [
        "dashboard-behavior.ts",
        "dashboard-style.html",
        "reload-behavior.ts",
        "run-color-style.html",
        "scrollbar-style.html",
        "tensorboard-color.html",
        "tf-categorizer.html",
        "tf-categorizer.ts",
        "tf-chart-scaffold.html",
        "tf-collapsable-pane.html",
        "tf-dashboard.html",
        "tf-dashboard-layout.html",
        "tf-downloader.html",
        "tf-multi-checkbox.html",
        "tf-multi-checkbox.ts",
        "tf-no-data-warning.html",
        "tf-option-selector.html",
        "tf-panes-helper.html",
        "tf-regex-group.html",
        "tf-regex-group.ts",
        "tf-run-selector.html",
        "tf-sidebar-helper.html",
    ],
    path = "/tf-dashboard-common",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "//tensorflow/tensorboard/components/tf_storage",
        "//tensorflow/tensorboard/components/vz_sorting",
        "@org_polymer_iron_ajax",
        "@org_polymer_iron_collapse",
        "@org_polymer_iron_icons",
        "@org_polymer_paper_button",
        "@org_polymer_paper_checkbox",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_input",
        "@org_polymer_paper_item",
        "@org_polymer_paper_menu",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_spinner",
        "@org_polymer_paper_styles",
        "@org_polymer_paper_toggle_button",
    ],
)

ts_web_library(
    name = "demo",
    srcs = [
        "tf-categorizer-demo.html",
        "tf-collapsable-pane-demo.html",
        "tf-multi-checkbox-demo.html",
        "tf-regex-group-demo.html",
    ],
    path = "/tf-dashboard-common",
    deps = [
        ":tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_color_scale",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_styles",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_dashboard_common"],
    destdir = "tf-dashboard-common",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports_google:lib",
        "//tensorflow/tensorboard/components/tf_storage:legacy",
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
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

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
