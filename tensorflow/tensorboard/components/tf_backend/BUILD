package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

# TODO(dandelion): Add webfiles support for the test code.

webfiles(
    name = "tf_backend",
    srcs = [
        "tf-backend.html",
        ":ts",
    ],
    path = "/tf-backend",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/vz_sorting",
        "@org_polymer",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = [
        "backend.ts",
        "behavior.ts",
        "requestManager.ts",
        "router.ts",
        "urlPathHelpers.ts",
    ],
    typings = [
        "@org_definitelytyped//:d3.d.ts",
        "@org_definitelytyped//:lodash.d.ts",
        "//tensorflow/tensorboard/components/vz_sorting:ts_typings",
    ],
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
        "tf-backend.html",
        ":legacy_ts",
    ],
    visibility = ["//visibility:public"],
    destdir = "tf-backend",
    deps = [
        "//tensorflow/tensorboard/components:tf_imports",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [
        "backend.ts",
        "behavior.ts",
        "requestManager.ts",
        "router.ts",
        "urlPathHelpers.ts",
    ],
    deps = [
        "//tensorflow/tensorboard/components:common_deps",
        "//tensorflow/tensorboard/components/vz_sorting:legacy_ts",
    ],
)
