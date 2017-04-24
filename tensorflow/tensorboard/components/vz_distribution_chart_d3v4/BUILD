package(default_visibility = ["//tensorflow:internal"])

load(
    "//javascript/typescript:build_defs.bzl",
    "ts_development_sources",
    "ts_library",
    "ts_devserver",
)
load(
    "//third_party/javascript/polymer/build_defs:build_defs.bzl",
    "webcomponent_library",
)

licenses(["notice"])  # Apache 2.0

ts_library(
    name = "ts",
    srcs = [
        "vz-distribution-chart.ts",
    ],
    deps = [
        "//tensorflow/tensorboard/components:common_deps_d3v4",
        "//tensorflow/tensorboard/components/vz_line_chart_d3v4:ts",
        "//third_party/javascript/lodash:lodash-module",
        "//third_party/javascript/plottable/v3:bundle",
    ],
)

webcomponent_library(
    name = "vz_distribution_chart",
    srcs = [
        "demo.html",
        "vz-distribution-chart.html",
    ],
    ts_lib_deps = [":ts"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
        "//third_party/javascript/plottable/v3:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

webcomponent_library(
    name = "demo",
    srcs = ["demo.html"],
    visibility = ["//visibility:public"],
    deps = [
        ":vz_distribution_chart",
        "//third_party/javascript/polymer/v1/iron-demo-helpers:lib",
        "//third_party/javascript/polymer/v1/paper-styles:lib",
    ],
)

ts_devserver(
    name = "devserver",
    manifest = ":dev_sources",
    serving_path = "/demo_out/bundle.js",
    static_files = [":demo"],
)

ts_development_sources(
    name = "dev_sources",
    deps = [
        ":ts",
    ],
)
