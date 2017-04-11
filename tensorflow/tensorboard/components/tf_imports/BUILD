package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "d3",
    srcs = [
        "d3.html",
        "@org_d3js",
    ],
    path = "/tf-imports",
)

webfiles(
    name = "lodash",
    srcs = [
        "lodash.html",
        "@com_lodash",
    ],
    path = "/tf-imports",
)

webfiles(
    name = "graphlib",
    srcs = [
        "graphlib.html",
        "@io_github_cpettitt_graphlib",
    ],
    path = "/tf-imports",
    deps = [":lodash"],
)

webfiles(
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

webfiles(
    name = "plottable",
    srcs = [
        "plottable.html",
        "@com_palantir_plottable//:plottable.css",
        "@com_palantir_plottable//:plottable.js",
    ],
    path = "/tf-imports",
    deps = [":d3"],
)
