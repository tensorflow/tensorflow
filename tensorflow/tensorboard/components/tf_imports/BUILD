package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "d3",
    srcs = [
        "d3.html",
        "@org_d3js",
    ],
    path = "/tf-imports",
)

web_library(
    name = "lodash",
    srcs = ["lodash.html"],
    path = "/tf-imports",
    deps = ["@com_lodash"],
)

web_library(
    name = "graphlib",
    srcs = [
        "graphlib.html",
        "@io_github_cpettitt_graphlib",
    ],
    path = "/tf-imports",
    deps = [":lodash"],
)

web_library(
    name = "dagre",
    srcs = [
        "dagre.html",
        "@io_github_cpettitt_dagre",
    ],
    path = "/tf-imports",
    deps = [
        ":graphlib",
        ":lodash",
    ],
)

web_library(
    name = "plottable",
    srcs = [
        "plottable.html",
        "@com_palantir_plottable//:plottable.css",
        "@com_palantir_plottable//:plottable.js",
    ],
    path = "/tf-imports",
    deps = [":d3"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
