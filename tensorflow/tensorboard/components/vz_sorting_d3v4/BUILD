package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:hacks.bzl", "tensorboard_typescript_bundle")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "vz_sorting_d3v4",
    srcs = [
        "bundle.js",
        "vz-sorting.html",
    ],
    path = "/vz-sorting",
    visibility = ["//visibility:public"],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["bundle.ts"],
)

tensorboard_typescript_bundle(
    name = "bundle",
    out = "bundle.ts",
    namespace_srcs = {"VZ.Sorting": ["sorting.ts"]},
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
