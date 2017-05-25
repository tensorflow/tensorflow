package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

web_library(
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
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "@org_polymer",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_spinner",
        "@org_polymer_paper_styles",
    ],
)

web_library(
    name = "index",
    srcs = [
        "demo/index.html",
        "index.html",
    ],
    path = "/tf-audio-dashboard",
    deps = [
        ":tf_audio_dashboard",
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/demo:demo_data",
        "@org_polymer_iron_component_page",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_styles",
        "@org_polymer_webcomponentsjs",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
