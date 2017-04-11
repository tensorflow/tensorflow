package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")

licenses(["notice"])  # Apache 2.0

# bazel run //third_party/tensorflow/tensorboard/components/tf_histogram_dashboard/demo
webfiles(
    name = "demo",
    srcs = ["index.html"],
    path = "/tf-histogram-dashboard/demo",
    deps = [
        "//tensorflow/tensorboard/components/tf_histogram_dashboard",
        "//tensorflow/tensorboard/components/tf_histogram_dashboard/demo/data",
        "//tensorflow/tensorboard/components/tf_imports:d3",
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
