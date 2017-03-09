package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

# TODO(dandelion): Add webfiles support for the test code.

webfiles(
    name = "tf_storage",
    srcs = [
        "tf-storage.html",
        ":ts",
    ],
    path = "/tf-storage",
    deps = [
        "//tensorflow/tensorboard/components/tf_globals",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "@org_polymer",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = [
        "storage.ts",
    ],
    typings = [
        "@org_definitelytyped//:lodash.d.ts",
        "//tensorflow/tensorboard/components/tf_globals:ts_typings",
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
        "tf-storage.html",
        ":legacy_ts",
    ],
    visibility = ["//visibility:public"],
    destdir = "tf-storage",
    deps = [
        "//tensorflow/tensorboard/components:tf_imports",
        "//tensorflow/tensorboard/components/tf_globals:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = [
        "storage.ts",
    ],
    deps = [
        "//tensorflow/tensorboard/components:common_deps",
        "//tensorflow/tensorboard/components/tf_globals:legacy_ts",
    ],
)
