package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "vz_histogram_timeseries",
    srcs = ["vz-histogram-timeseries.html"],
    path = "/vz-histogram-timeseries",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
    ],
)

ts_web_library(
    name = "demo",
    srcs = ["index.html"],
    path = "/vz-histogram-timeseries",
    deps = [
        ":vz_histogram_timeseries",
        "//tensorflow/tensorboard/components/tf_imports:webcomponentsjs",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_button",
        "@org_polymer_paper_styles",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":vz_histogram_timeseries"],
    visibility = ["//learning/vis/vz_elements/catalog:__pkg__"],
    destdir = "vz-histogram-timeseries",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports_google:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
