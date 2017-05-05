package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")

licenses(["notice"])  # Apache 2.0

# bazel run //third_party/tensorflow/tensorboard/components/vz_histogram_timeseries/demo
webfiles(
    name = "demo",
    srcs = ["index.html"],
    path = "/vz-histogram-timeseries/demo",
    deps = [
        "//tensorflow/tensorboard/components/vz_histogram_timeseries",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_button",
        "@org_polymer_paper_styles",
        "@org_polymer_webcomponentsjs",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
