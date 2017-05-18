package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

web_library(
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
    srcs = ["vz-distribution-chart.ts"],
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
        "//tensorflow/tensorboard/components/tf_imports_google:lib",
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = ["vz-distribution-chart.ts"],
    deps_mgmt = "off",
    runtime = "nodejs",
    deps = [
        "//tensorflow/tensorboard/components/vz_line_chart:legacy_ts",
        "//third_party/javascript/node_modules/typescript:es2015.promise",
        "//third_party/javascript/plottable/v1:typings",
        "//third_party/javascript/typings/chai",
        "//third_party/javascript/typings/d3",
        "//third_party/javascript/typings/lodash",
        "//third_party/javascript/typings/mocha",
        "//third_party/javascript/typings/polymer:polymer_without_externs",
        "//third_party/javascript/typings/sinon",
        "//third_party/javascript/typings/webcomponents_js",
    ],
)
