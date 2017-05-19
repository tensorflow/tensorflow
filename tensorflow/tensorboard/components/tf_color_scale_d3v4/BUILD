package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:hacks.bzl", "tensorboard_typescript_bundle")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "tf_color_scale_d3v4",
    srcs = [
        "bundle.js",
        "tf-color-scale.html",
    ],
    path = "/tf-color-scale",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports_d3v4:d3",
        "@org_polymer",
    ],
)

web_library(
    name = "demo",
    srcs = ["index.html"],
    path = "/tf-color-scale",
    deps = [
        ":tf_color_scale_d3v4",
        "//tensorflow/tensorboard/components/tf_imports_d3v4:d3",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_button",
        "@org_polymer_paper_styles",
        "@org_polymer_webcomponentsjs",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["bundle.ts"],
    typings = [
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
        "//tensorflow/tensorboard/components/tf_imports_d3v4:d3.d.ts",
    ],
)

tensorboard_typescript_bundle(
    name = "bundle",
    out = "bundle.ts",
    namespace_srcs = {"TF": [
        "palettes.ts",
        "colorScale.ts",
    ]},
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
