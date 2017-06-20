package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "vz_sorting",
    srcs = [
        "sorting.ts",
        "vz-sorting.html",
    ],
    path = "/vz-sorting",
    visibility = ["//visibility:public"],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":vz_sorting"],
    destdir = "vz-sorting",
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
