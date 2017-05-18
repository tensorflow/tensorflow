package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:vulcanize.bzl", "tensorboard_html_binary")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_tensorboard_d3v4",
    srcs = [
        "tf-tensorboard.html",
        ":ts",
    ],
    path = "/tf-tensorboard",
    deps = [
        "@org_polymer_iron_icons",
        "@org_polymer_paper_button",
        "@org_polymer_paper_checkbox",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_header_panel",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_tabs",
        "@org_polymer_paper_toolbar",
        "@org_polymer",
        "//tensorflow/tensorboard/components/tf_audio_dashboard_d3v4",
        "//tensorflow/tensorboard/components/tf_backend_d3v4",
        "//tensorflow/tensorboard/components/tf_dashboard_common_d3v4",
        "//tensorflow/tensorboard/components/tf_distribution_dashboard_d3v4",
        "//tensorflow/tensorboard/components/tf_globals_d3v4",
        # "//tensorflow/tensorboard/components/tf_graph_dashboard_d3v4",
        "//tensorflow/tensorboard/components/tf_histogram_dashboard_d3v4",
        "//tensorflow/tensorboard/components/tf_image_dashboard_d3v4",
        "//tensorflow/tensorboard/components/tf_scalar_dashboard_d3v4",
        "//tensorflow/tensorboard/components/tf_storage_d3v4",
        "//tensorflow/tensorboard/components/tf_text_dashboard_d3v4",
        "//tensorflow/tensorboard/components/vz_projector_d3v4",
    ],
)

web_library(
    name = "demo",
    srcs = ["demo.html"],
    path = "/tf-tensorboard",
    deps = [
        ":tf_tensorboard_d3v4",
        "//tensorflow/tensorboard/demo:demo_data",
        "@org_polymer_webcomponentsjs",
    ],
)

web_library(
    name = "dist",
    srcs = ["dist.html"],
    path = "/tf-tensorboard",
    deps = [
        ":tf_tensorboard_d3v4",
        "//tensorflow/tensorboard/demo:demo_data",
        "@org_polymer_webcomponentsjs",
    ],
)

tensorboard_html_binary(
    name = "index",
    # input_path = "/tf-dashboard-common/tf-chart-scaffold.html",
    input_path = "/tf-tensorboard/dist.html",
    output_path = "/index.html",
    deps = [":dist"],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["autoReloadBehavior.ts"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
