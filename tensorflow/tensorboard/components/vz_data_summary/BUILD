package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow/tensorboard:defs.bzl",
    "tensorboard_ts_config",
    "tensorboard_ts_declaration",
    "tensorboard_ts_development_sources",
    "tensorboard_ts_devserver",
    "tensorboard_ts_library",
    "tensorboard_webcomponent_library",
)

licenses(["notice"])  # Apache 2.0

tensorboard_webcomponent_library(
    name = "lib",
    srcs = ["vz-data-summary.html"],
    ts_lib_deps = [":ts_lib"],
    destdir = "vz-data-summary",
    deps = [
        "//learning/vis/vz_elements:common",
        "//third_party/javascript/d3/v3:lib",
        "//third_party/javascript/polymer/v1/iron-demo-helpers:lib",
        "//third_party/javascript/polymer/v1/iron-resizable-behavior:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "ts_lib",
    srcs = ["vz-data-summary.ts"],
    externs_list = [":externs"],
    deps = [
        ":typings",
        "//third_party/javascript/typings/polymer:polymer_without_externs",
    ],
)

tensorboard_ts_declaration(
    name = "typings",
    srcs = ["typings.d.ts"],
)

# This build rule is used to run the demo.
tensorboard_ts_devserver(
    name = "dev_server",
    manifest = ":dev_sources",
    serving_path = "/demo_lib_out/vz-data-summary/vz-data-summary.js",
    static_files = [":demo_lib"],
    deps = [":tsconfig"],
)

tensorboard_webcomponent_library(
    name = "demo_lib",
    srcs = ["demo.html"],
    destdir = "vz-data-summary",
    deps = [
        ":lib",
        "//third_party/javascript/d3/v3:lib",
        "//third_party/javascript/polymer/v1/iron-demo-helpers:lib",
        "//third_party/javascript/polymer/v1/iron-resizable-behavior:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_ts_library(
    name = "demo_ts_lib",
    srcs = ["demo.ts"],
    externs_list = [":externs"],
    deps = [
        ":ts_lib",
        "//third_party/javascript/typings/d3",
    ],
)

tensorboard_ts_development_sources(
    name = "dev_sources",
    deps = [":demo_ts_lib"],
)

tensorboard_ts_config(
    name = "tsconfig",
    deps = [":ts_lib"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
    visibility = ["//tensorflow:__subpackages__"],
)
