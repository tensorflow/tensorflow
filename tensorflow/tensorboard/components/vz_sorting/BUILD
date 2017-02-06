package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "vz_sorting",
    srcs = [
        "vz-sorting.html",
        ":ts",
    ],
    path = "/vz-sorting",
    visibility = ["//visibility:public"],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["sorting.ts"],
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
        "vz-sorting.html",
        ":legacy_ts",
    ],
    visibility = ["//visibility:public"],
    destdir = "vz-sorting",
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = ["sorting.ts"],
    deps = ["//tensorflow/tensorboard/components:common_deps"],
)
