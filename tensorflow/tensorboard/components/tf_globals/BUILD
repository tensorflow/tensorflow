package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

# TODO(dandelion): Add webfiles support for the test code.

webfiles(
    name = "tf_globals",
    srcs = [
        "tf-globals.html",
        ":ts",
    ],
    path = "/tf-globals",
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["globals.ts"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)

################################################################################
# MARKED FOR DELETION

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [
        "tf-globals.html",
        ":legacy_ts",
    ],
    destdir = "tf-globals",
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [
        "globals.ts",
    ],
)
