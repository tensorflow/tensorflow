package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:hacks.bzl", "tensorboard_typescript_bundle")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_backend",
    srcs = [
        "bundle.js",
        "tf-backend.html",
    ],
    path = "/tf-backend",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_imports:plottable",
        "//tensorflow/tensorboard/components/vz_sorting",
        "@org_polymer",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["bundle.ts"],
    typings = [
        "@org_definitelytyped//:lodash.d.ts",
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
        "//tensorflow/tensorboard/components/tf_imports:d3.d.ts",
        "//tensorflow/tensorboard/components/tf_imports:plottable.d.ts",
        "//tensorflow/tensorboard/components/vz_sorting:bundle.d.ts",
    ],
)

tensorboard_typescript_bundle(
    name = "bundle",
    out = "bundle.ts",
    namespace_srcs = {"TF.Backend": [
        "requestManager.ts",
        "backend.ts",
        "behavior.ts",
        "urlPathHelpers.ts",
        "router.ts",
    ]},
    namespace_symbol_aliases = {"TF.Backend": {
        "compareTagNames": "VZ.Sorting.compareTagNames",
    }},
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
