package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:hacks.bzl", "tensorboard_typescript_bundle")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "vz_distribution_chart",
    srcs = [
        "bundle.js",
        "vz-distribution-chart.html",
    ],
    path = "/vz-distribution-chart",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_imports:plottable",
        "//tensorflow/tensorboard/components/vz_line_chart",
        "@org_polymer",
    ],
)

web_library(
    name = "demo",
    srcs = ["index.html"],
    path = "/vz-distribution-chart",
    deps = [
        ":vz_distribution_chart",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_styles",
        "@org_polymer_webcomponentsjs",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["bundle.ts"],
    typings = [
        "//tensorflow/tensorboard/components/tf_imports:d3.d.ts",
        "//tensorflow/tensorboard/components/tf_imports:plottable.d.ts",
        "@org_definitelytyped//:lodash.d.ts",
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
        "//tensorflow/tensorboard/components/vz_line_chart:bundle.d.ts",
    ],
)

tensorboard_typescript_bundle(
    name = "bundle",
    out = "bundle.ts",
    namespace_srcs = {"VZ": [
        "vz-distribution-chart.ts",
    ]},
    namespace_symbol_aliases = {"VZ": {
        "Dataset": "Plottable.Dataset",
        "ChartHelpers": "VZ.ChartHelpers",
    }},
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
