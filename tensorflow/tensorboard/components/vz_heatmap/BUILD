package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [
        "demo/index.html",
        "index.html",
        "vz-heatmap.html",
    ],
    visibility = ["//visibility:public"],
    destdir = "vz-heatmap",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports_google:lib",
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
        "//third_party/javascript/polymer/v1/iron-component-page:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
        "//third_party/javascript/polymer/v1/webcomponentsjs:lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
