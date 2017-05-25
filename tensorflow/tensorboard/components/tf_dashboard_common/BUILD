package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:hacks.bzl", "tensorboard_typescript_bundle")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_dashboard_common",
    srcs = [
        "dashboard-style.html",
        "run-color-style.html",
        "scrollbar-style.html",
        "tensorboard-color.html",
        "tf-categorizer.html",
        "tf-categorizer-bundle.js",
        "tf-chart-scaffold.html",
        "tf-collapsable-pane.html",
        "tf-dashboard.html",
        "tf-dashboard.js",
        "tf-dashboard-layout.html",
        "tf-downloader.html",
        "tf-multi-checkbox.html",
        "tf-multi-checkbox-bundle.js",
        "tf-no-data-warning.html",
        "tf-option-selector.html",
        "tf-panes-helper.html",
        "tf-regex-group.html",
        "tf-regex-group-bundle.js",
        "tf-run-selector.html",
        "tf-sidebar-helper.html",
    ],
    path = "/tf-dashboard-common",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_storage",
        "//tensorflow/tensorboard/components/vz_sorting",
        "@org_polymer",
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

web_library(
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

tensorboard_typescript_bundle(
    name = "tf_categorizer_bundle",
    out = "tf-categorizer-bundle.ts",
    namespace_srcs = {"TF.Dashboard.Categorizer": ["tf-categorizer.ts"]},
    namespace_symbol_aliases = {"TF.Dashboard.Categorizer": {"compareTagNames": "VZ.Sorting.compareTagNames"}},
)

tensorboard_typescript_genrule(
    name = "tf_categorizer_ts",
    srcs = ["tf-categorizer-bundle.ts"],
    typings = [
        "@org_definitelytyped//:lodash.d.ts",
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
        "//tensorflow/tensorboard/components/tf_imports:d3.d.ts",
        "//tensorflow/tensorboard/components/vz_sorting:bundle.d.ts",
    ],
)

tensorboard_typescript_bundle(
    name = "tf_regex_group_bundle",
    out = "tf-regex-group-bundle.ts",
    namespace_srcs = {"TF.Dashboard.RegexGroup": ["tf-regex-group.ts"]},
    namespace_symbol_aliases = {"TF.Dashboard.RegexGroup": {"storage": "TF.URIStorage"}},
)

tensorboard_typescript_genrule(
    name = "tf_regex_group_ts",
    srcs = ["tf-regex-group-bundle.ts"],
    typings = [
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
        "//tensorflow/tensorboard/components/tf_storage:bundle.d.ts",
    ],
)

tensorboard_typescript_bundle(
    name = "tf_multi_checkbox_bundle",
    out = "tf-multi-checkbox-bundle.ts",
    namespace_srcs = {"TF.Dashboard.MultiCheckbox": ["tf-multi-checkbox.ts"]},
    namespace_symbol_aliases = {"TF.Dashboard.MultiCheckbox": {"storage": "TF.URIStorage"}},
)

tensorboard_typescript_genrule(
    name = "tf_multi_checkbox_ts",
    srcs = ["tf-multi-checkbox-bundle.ts"],
    typings = [
        "@org_definitelytyped//:lodash.d.ts",
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
        "//tensorflow/tensorboard/components/tf_storage:bundle.d.ts",
    ],
)

tensorboard_typescript_bundle(
    name = "tf_dashboard_bundle",
    out = "tf-dashboard.ts",
    namespace_srcs = {
        "TF.Dashboard": [
            "dashboard-behavior.ts",
            "reload-behavior.ts",
        ],
    },
)

tensorboard_typescript_genrule(
    name = "tf_dashboard_ts",
    srcs = ["tf-dashboard.ts"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)

################################################################################
# MARKED FOR DELETION

load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = glob(["*.html"]) + [":legacy_ts"],
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

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [
        "dashboard-behavior.ts",
        "reload-behavior.ts",
        "tf-categorizer.ts",
    ],
    deps_mgmt = "off",
    runtime = "nodejs",
    deps = [
        "//tensorflow/tensorboard/components/vz_sorting:legacy_ts",
        "//third_party/javascript/typings/d3_v4:bundle",
        "//third_party/javascript/typings/lodash",
        "//third_party/javascript/typings/polymer:polymer_without_externs",
    ],
)
