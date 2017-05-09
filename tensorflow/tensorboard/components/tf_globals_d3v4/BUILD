package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")

licenses(["notice"])  # Apache 2.0

tensorboard_ts_library(
    name = "ts",
    srcs = ["globals.ts"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
