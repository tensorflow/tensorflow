package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_tensorboard",
    srcs = [
        "style.html",
        "tf-tensorboard.html",
        ":ts",
    ],
    path = "/tf-tensorboard",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tensorboard/components/tf_audio_dashboard",
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_dashboard_common",
        "//tensorflow/tensorboard/components/tf_distribution_dashboard",
        "//tensorflow/tensorboard/components/tf_globals",
        "//tensorflow/tensorboard/components/tf_graph_dashboard",
        "//tensorflow/tensorboard/components/tf_histogram_dashboard",
        "//tensorflow/tensorboard/components/tf_image_dashboard",
        "//tensorflow/tensorboard/components/tf_scalar_dashboard",
        "//tensorflow/tensorboard/components/tf_storage",
        "//tensorflow/tensorboard/components/tf_text_dashboard",
        "//tensorflow/tensorboard/components/vz_projector",
        "@org_polymer",
        "@org_polymer_font_roboto",
        "@org_polymer_iron_icons",
        "@org_polymer_paper_button",
        "@org_polymer_paper_checkbox",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_header_panel",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_tabs",
        "@org_polymer_paper_toolbar",
    ],
)

web_library(
    name = "demo",
    srcs = ["demo.html"],
    path = "/tf-tensorboard",
    deps = [
        ":tf_tensorboard",
        "//tensorflow/tensorboard/demo:demo_data",
        "@org_polymer_webcomponentsjs",
    ],
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
