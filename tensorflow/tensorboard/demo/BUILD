package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

# THIS PACKAGE HAS MOVED
# See tensorflow/tensorboard/components/tf_tensorboard:demo

web_library(
    name = "demo_data",
    srcs = glob(["data/**"]),
    path = "/",
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
