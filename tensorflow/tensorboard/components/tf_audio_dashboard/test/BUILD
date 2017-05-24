package(
    default_testonly = True,
    default_visibility = ["//tensorflow:internal"],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "test",
    srcs = [
        "audioDashboardTests.js",
        "tests.html",
    ],
    path = "/tf-audio-dashboard/test",
    deps = [
        "//tensorflow/tensorboard/components/tf_audio_dashboard",
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/demo:demo_data",
        "@org_npmjs_registry_web_component_tester",
        "@org_polymer",
        "@org_polymer_webcomponentsjs",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = ["audioDashboardTests.ts"],
    typings = [
        "@org_definitelytyped//:chai.d.ts",
        "@org_definitelytyped//:mocha.d.ts",
        "@org_definitelytyped//:sinon.d.ts",
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
        "//tensorflow/tensorboard/components/tf_backend:bundle.d.ts",
        "//tensorflow/tensorboard/components/tf_imports:d3.d.ts",
    ],
)

filegroup(
    name = "all_files",
    testonly = 0,
    srcs = glob(["**"]),
    tags = ["notsan"],
)
