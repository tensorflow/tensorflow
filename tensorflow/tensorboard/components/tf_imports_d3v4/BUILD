package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "d3",
    srcs = [
        "d3.html",
        "//third_party/javascript/d3/v4:d3.js",
    ],
    path = "/tf-imports-d3v4",
)

webfiles(
    name = "lodash",
    srcs = [
        "lodash.html",
        "@com_lodash",
    ],
    path = "/tf-imports-d3v4",
)

webfiles(
    name = "graphlib",
    srcs = [
        "graphlib.html",
        "@io_github_cpettitt_graphlib",
    ],
    path = "/tf-imports-d3v4",
    deps = [":lodash"],
)

webfiles(
    name = "dagre",
    srcs = [
        "dagre.html",
        "@io_github_cpettitt_dagre",
    ],
    path = "/tf-imports-d3v4",
    deps = [
        ":graphlib",
        ":lodash",
    ],
)

webfiles(
    name = "plottable",
    srcs = [
        "plottable.html",
        "//third_party/javascript/plottable/v3:plottable.css",
        "//third_party/javascript/plottable/v3:plottable.js",
    ],
    path = "/tf-imports-d3v4",
    deps = [":d3"],
)
