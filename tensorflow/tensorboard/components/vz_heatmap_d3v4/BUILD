package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "vz_heatmap_d3v4",
    srcs = ["vz-heatmap.html"],
    path = "/vz-heatmap",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tensorboard/components/tf_imports_d3v4:d3",
        "@org_polymer",
        "@org_polymer_iron_resizable_behavior",
    ],
)

web_library(
    name = "index",
    srcs = [
        "demo/index.html",
        "index.html",
    ],
    path = "/vz-heatmap",
    visibility = ["//visibility:public"],
    deps = [
        ":vz_heatmap_d3v4",
        "//tensorflow/tensorboard/components/tf_imports_d3v4:d3",
        "@org_polymer",
        "@org_polymer_iron_component_page",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_webcomponentsjs",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
