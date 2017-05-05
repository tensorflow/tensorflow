package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "tf_graph_common",
    srcs = [
        "tf-graph-common.html",
        ":ts",
    ],
    path = "/tf-graph-common",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:dagre",
        "//tensorflow/tensorboard/components/tf_imports:graphlib",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "@org_polymer",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = glob(["*.ts"]),
    typings = [
        "@org_definitelytyped//:d3.d.ts",
        "@org_definitelytyped//:lodash.d.ts",
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
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
        "tf-graph-common.html",
        ":legacy_ts",
    ],
    destdir = "tf-graph-common",
    deps = [
        "//tensorflow/tensorboard/components:tf_imports",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "legacy_ts",
    srcs = glob(["*.ts"]),
    deps_mgmt = "off",
    runtime = "nodejs",
    deps = ["//tensorflow/tensorboard/components:common_deps"],
)
