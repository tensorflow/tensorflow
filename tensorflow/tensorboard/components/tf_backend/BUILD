package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "tf_backend",
    srcs = [
        "backend.ts",
        "behavior.ts",
        "requestManager.ts",
        "router.ts",
        "runsStore.ts",
        "tf-backend.html",
        "urlPathHelpers.ts",
    ],
    path = "/tf-backend",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
        "//tensorflow/tensorboard/components/tf_imports:plottable",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "//tensorflow/tensorboard/components/vz_sorting",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_backend"],
    destdir = "tf-backend",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports_google:lib",
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
