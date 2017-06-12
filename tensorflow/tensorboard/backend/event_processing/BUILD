# Description:
# Event processing logic for TensorBoard

package(default_visibility = ["//tensorflow/tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "io_wrapper",
    srcs = ["io_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "directory_watcher",
    srcs = ["directory_watcher.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":io_wrapper",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "directory_watcher_test",
    size = "small",
    srcs = ["directory_watcher_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":directory_watcher",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "reservoir",
    srcs = ["reservoir.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "reservoir_test",
    size = "small",
    srcs = ["reservoir_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":reservoir",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "event_file_loader",
    srcs = ["event_file_loader.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "event_file_loader_test",
    size = "small",
    srcs = ["event_file_loader_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":event_file_loader",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "event_accumulator",
    srcs = ["event_accumulator.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":directory_watcher",
        ":event_file_loader",
        ":plugin_asset_util",
        ":reservoir",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "event_accumulator_test",
    size = "small",
    srcs = ["event_accumulator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":event_accumulator",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "event_multiplexer",
    srcs = ["event_multiplexer.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":directory_watcher",
        ":event_accumulator",
        ":io_wrapper",
        "//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "event_multiplexer_test",
    size = "small",
    srcs = ["event_multiplexer_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":event_accumulator",
        ":event_multiplexer",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "plugin_asset_util",
    srcs = ["plugin_asset_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "event_file_inspector",
    srcs = ["event_file_inspector.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":event_accumulator",
        ":event_multiplexer",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "event_file_inspector_test",
    size = "small",
    srcs = ["event_file_inspector_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":event_file_inspector",
        "//tensorflow:tensorflow_py",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        [
            "*",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
