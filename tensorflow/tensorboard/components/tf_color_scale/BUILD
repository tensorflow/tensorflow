package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

# TODO(dandelion): Add webfiles support for the test code.

webfiles(
    name = "tf_color_scale",
    srcs = [
        "tf-color-scale.html",
        ":ts",
    ],
    path = "/tf-color-scale",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "@org_polymer",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = [
        "colorScale.ts",
        "palettes.ts",
    ],
    typings = ["@org_definitelytyped//:d3.d.ts"],
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
        "tf-color-scale.html",
        ":legacy_ts",
    ],
    destdir = "tf-color-scale",
    deps = [
        "//tensorflow/tensorboard/components:tf_imports",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [
        "colorScale.ts",
        "palettes.ts",
    ],
    deps = ["//tensorflow/tensorboard/components:common_deps"],
)
