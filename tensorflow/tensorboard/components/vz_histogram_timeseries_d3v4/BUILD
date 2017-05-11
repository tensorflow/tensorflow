package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "vz_histogram_timeseries",
    srcs = [
        "vz-histogram-timeseries.html",
    ],
    path = "/vz-histogram-timeseries",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports_d3v4:d3",
        "@org_polymer",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)

# bazel run //third_party/tensorflow/tensorboard/components/vz_histogram_timeseries/demo
web_library(
    name = "demo",
    srcs = ["demo.html"],
    path = "/vz-histogram-timeseries",
    deps = [
        ":vz_histogram_timeseries",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_button",
        "@org_polymer_paper_styles",
        "@org_polymer_webcomponentsjs",
    ],
)
