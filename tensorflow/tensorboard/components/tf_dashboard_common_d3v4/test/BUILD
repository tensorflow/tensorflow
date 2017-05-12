package(
    default_testonly = True,
    default_visibility = ["//tensorflow:internal"],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:hacks.bzl", "tensorboard_typescript_bundle")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "test",
    srcs = [
        "bundle.js",
        "tests.html",
    ],
    path = "/tf-dashboard-common/test",
    deps = [
        "//tensorflow/tensorboard/components/tf_dashboard_common_d3v4",
        "@org_npmjs_registry_web_component_tester",
        "@org_polymer",
        "@org_polymer_webcomponentsjs",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["bundle.ts"],
    typings = [
        "@org_definitelytyped//:chai.d.ts",
        "@org_definitelytyped//:mocha.d.ts",
        "//tensorflow/tensorboard/components/tf_dashboard_common_d3v4:bundle.d.ts",
    ],
)

tensorboard_typescript_bundle(
    name = "bundle",
    out = "bundle.ts",
    namespace_srcs = {"TF.Dashboard": ["tf-categorizer-tests.ts"]},
)

filegroup(
    name = "all_files",
    testonly = 0,
    srcs = glob(["**"]),
    tags = ["notsan"],
)
