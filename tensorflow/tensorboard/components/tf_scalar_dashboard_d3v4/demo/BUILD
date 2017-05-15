package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "demo",
    srcs = ["index.html"],
    path = "/tf-scalar-dashboard/demo",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend_d3v4",
        "//tensorflow/tensorboard/components/tf_imports_d3v4:d3",
        "//tensorflow/tensorboard/components/tf_scalar_dashboard_d3v4",
        "//tensorflow/tensorboard/demo:demo_data",
        "@org_polymer",
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
