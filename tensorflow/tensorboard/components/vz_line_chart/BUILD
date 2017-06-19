package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "vz_line_chart",
    srcs = [
        "dragZoomInteraction.ts",
        "vz-chart-helpers.ts",
        "vz-line-chart.html",
        "vz-line-chart.ts",
    ],
    path = "/vz-line-chart",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_imports:plottable",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
    ],
)

ts_web_library(
    name = "demo",
    srcs = ["index.html"],
    path = "/vz-line-chart",
    deps = [
        ":vz_line_chart",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_styles",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":vz_line_chart"],
    visibility = ["//learning/vis/vz_elements/catalog:__pkg__"],
    destdir = "vz-line-chart",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports_google:lib",
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
