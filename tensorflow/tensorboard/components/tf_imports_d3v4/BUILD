package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "d3",
    srcs = [
        "d3.html",
        "@org_d3js_v4",
    ],
    path = "/tf-imports-d3v4",
)

webfiles(
    name = "lodash",
    srcs = ["lodash.html"],
    path = "/tf-imports-d3v4",
    deps = ["@com_lodash"],
)

webfiles(
    name = "threejs",
    srcs = [
        "threejs.html",
        "@org_threejs//:OrbitControls.js",
        "@org_threejs//:three.js",
    ],
    path = "/tf-imports",
)

webfiles(
    name = "numericjs",
    srcs = [
        "numericjs.html",
        "@com_numericjs",
    ],
    path = "/tf-imports",
)

webfiles(
    name = "weblas",
    srcs = [
        "weblas.html",
        "@io_github_waylonflinn_weblas",
    ],
    path = "/tf-imports",
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
        # TODO(jart): Replace when Plottable v3 is fixed.
        "plottable.js",
        "plottable.css",
        # "//third_party/javascript/plottable/v3:plottable.css",
        # "//third_party/javascript/plottable/v3:plottable.js",
    ],
    path = "/tf-imports-d3v4",
    deps = [":d3"],
)

genrule(
    name = "TEMPORARY_fake_plottable_css",
    outs = ["plottable.css"],
    cmd = "echo >$@",
)

genrule(
    name = "TEMPORARY_fake_plottable_js",
    outs = ["plottable.js"],
    cmd = "echo >$@",
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
