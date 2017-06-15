package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")
load("//tensorflow/tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "tensorboard",
    srcs = [
        "analytics.html",
        "tensorboard.html",
    ],
    path = "/",
    deps = ["//tensorflow/tensorboard/components/tf_tensorboard"],
)

tensorboard_html_binary(
    name = "index",
    input_path = "/tensorboard.html",
    output_path = "/index.html",
    deps = [":tensorboard"],
)

ts_web_library(
    name = "trace_viewer",
    srcs = [
        "trace_viewer.html",
    ],
    path = "/",
    deps = [
        "//tensorflow/tensorboard/components/tf_trace_viewer",
    ],
)

tensorboard_html_binary(
    name = "trace_viewer_index",
    input_path = "/trace_viewer.html",
    output_path = "/trace_viewer_index.html",
    deps = [":trace_viewer"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
