package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "vz_line_chart",
    srcs = [
        "vz-line-chart.html",
        ":ts",
    ],
    path = "/vz-line-chart",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_imports:plottable",
        "@org_polymer",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = [
        "dragZoomInteraction.ts",
        "vz-chart-helpers.ts",
        "vz-line-chart.ts",
    ],
    typings = [
        "@org_definitelytyped//:d3.d.ts",
        "@com_palantir_plottable//:plottable.d.ts",
        "@org_definitelytyped//:lodash.d.ts",
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
        "index.html",
        "vz-line-chart.html",
        ":legacy_ts",
    ],
    visibility = ["//visibility:public"],
    destdir = "vz-line-chart",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports_google:lib",
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [
        "dragZoomInteraction.ts",
        "vz-chart-helpers.ts",
        "vz-line-chart.ts",
    ],
    deps_mgmt = "off",
    runtime = "nodejs",
    deps = [
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
