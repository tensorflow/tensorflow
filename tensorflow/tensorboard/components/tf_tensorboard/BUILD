package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")
load("//tensorflow/tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "tf_tensorboard",
    srcs = [
        "autoReloadBehavior.ts",
        "style.html",
        "tf-tensorboard.html",
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
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "//tensorflow/tensorboard/components/tf_scalar_dashboard",
        "//tensorflow/tensorboard/components/tf_storage",
        "//tensorflow/tensorboard/components/tf_text_dashboard",
        "//tensorflow/tensorboard/components/vz_projector",
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

ts_web_library(
    name = "demo",
    srcs = ["demo.html"],
    path = "/tf-tensorboard",
    deps = [
        ":tf_tensorboard",
        "//tensorflow/tensorboard/demo:demo_data",
    ],
)

tensorboard_html_binary(
    name = "devserver",
    testonly = 1,
    input_path = "/tf-tensorboard/demo.html",
    output_path = "/index.html",
    deps = [":demo"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
