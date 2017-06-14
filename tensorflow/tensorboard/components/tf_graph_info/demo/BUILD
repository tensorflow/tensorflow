package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

# bazel run //third_party/tensorflow/tensorboard/components/tf_graph_info/demo
ts_web_library(
    name = "demo",
    srcs = ["index.html"] + glob(["data/**"]),
    path = "/tf-graph-info/demo",
    deps = [
        "//tensorflow/tensorboard/components/tf_graph_common",
        "//tensorflow/tensorboard/components/tf_graph_info",
        "//tensorflow/tensorboard/components/tf_graph_loader",
        "//tensorflow/tensorboard/components/tf_imports:webcomponentsjs",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_styles",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
