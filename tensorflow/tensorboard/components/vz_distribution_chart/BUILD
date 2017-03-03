package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "vz_distribution_chart",
    srcs = [
        "vz-distribution-chart.html",
        ":ts",
    ],
    path = "/vz-distribution-chart",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_imports:plottable",
        "//tensorflow/tensorboard/components/vz_line_chart",
        "@org_polymer",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = [
        "vz-distribution-chart.ts",
    ],
    typings = [
        "@org_definitelytyped//:d3.d.ts",
        "@com_palantir_plottable//:plottable.d.ts",
        "@org_definitelytyped//:lodash.d.ts",
        "//tensorflow/tensorboard/components/vz_line_chart:ts_typings",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)

################################################################################
# MARKED FOR DELETION

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [
        "vz-distribution-chart.html",
        ":legacy_ts",
    ],
    visibility = ["//visibility:public"],
    destdir = "vz-distribution-chart",
    deps = [
        "//tensorflow/tensorboard/components:tf_imports",
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [
        "vz-distribution-chart.ts",
    ],
    deps = [
        "//tensorflow/tensorboard/components:common_deps",
        "//tensorflow/tensorboard/components/vz_line_chart:legacy_ts",
    ],
)
