package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

# bazel run //third_party/tensorflow/tensorboard/components/tf_graph_loader/demo
web_library(
    name = "demo",
    srcs = ["index.html"] + glob(["data/**"]),
    path = "/tf-graph-loader/demo",
    deps = [
        "//tensorflow/tensorboard/components/tf_graph_loader_d3v4",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_styles",
        "@org_polymer_webcomponentsjs",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
