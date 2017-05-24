package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:vulcanize.bzl", "tensorboard_html_binary")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tensorboard",
    srcs = [
        "analytics.html",
        "tensorboard.html",
    ],
    path = "/",
    deps = [
        "//tensorflow/tensorboard/components/tf_tensorboard",
        "@org_polymer_webcomponentsjs",
    ],
)

tensorboard_html_binary(
    name = "index",
    input_path = "/tensorboard.html",
    output_path = "/index.html",
    deps = [":tensorboard"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
