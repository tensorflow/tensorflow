package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:hacks.bzl", "tensorboard_typescript_bundle")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_globals_d3v4",
    srcs = [
        "bundle.js",
        "tf-globals.html",
    ],
    path = "/tf-globals",
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["bundle.ts"],
)

tensorboard_typescript_bundle(
    name = "bundle",
    out = "bundle.ts",
    namespace_srcs = {"TF.Globals": ["globals.ts"]},
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
