package(default_visibility = ["//tensorflow/tensorboard:internal"])

load("//tensorflow/tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "tf_storage",
    srcs = [
        "storage.ts",
        "tf-storage.html",
    ],
    path = "/tf-storage",
    deps = [
        "//tensorflow/tensorboard/components/tf_globals",
        "//tensorflow/tensorboard/components/tf_imports:lodash",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_storage"],
    destdir = "tf-storage",
    deps = [
        "//tensorflow/tensorboard/components/tf_globals:legacy",
        "//tensorflow/tensorboard/components/tf_imports_google:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
