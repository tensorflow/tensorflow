# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

package_group(
    name = "internal",
    packages = [
        "//learning/brain/tensorboard/...",
        "//learning/vis/...",
        "//tensorflow/...",
        "//tensorflow/tensorboard/...",
    ],
)

py_binary(
    name = "tensorboard",
    srcs = ["main.py"],
    data = [":assets"],
    main = "main.py",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/tensorboard/backend:application",
        "//tensorflow/tensorboard/backend/event_processing:event_file_inspector",
        "//tensorflow/tensorboard/plugins/audio:audio_plugin",
        "//tensorflow/tensorboard/plugins/distributions:distributions_plugin",
        "//tensorflow/tensorboard/plugins/graphs:graphs_plugin",
        "//tensorflow/tensorboard/plugins/histograms:histograms_plugin",
        "//tensorflow/tensorboard/plugins/images:images_plugin",
        "//tensorflow/tensorboard/plugins/projector:projector_plugin",
        "//tensorflow/tensorboard/plugins/scalars:scalars_plugin",
        "//tensorflow/tensorboard/plugins/text:text_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
    ],
)

py_library(
    name = "expect_tensorflow_installed",
    # This is a dummy rule used as a TensorFlow dependency in open-source.
    # We expect TensorFlow to already be installed on the system, e.g. via
    # `pip install tensorflow`
)

py_library(
    name = "expect_numpy_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect numpy to already be installed on the system, e.g. via
    # `pip install numpy`
)

filegroup(
    name = "assets",
    srcs = [
        "TAG",
        "//tensorflow/tensorboard/components:index.html",
        "//tensorflow/tensorboard/components:trace_viewer_index.html",
    ],
)

filegroup(
    name = "ts_web_library_default_typings",
    srcs = [
        # Ordering probably matters.
        "@com_microsoft_typescript//:lib.es6.d.ts",
        "@io_angular_clutz//:src/resources/closure.lib.d.ts",
        "//tensorflow/tensorboard/defs:clutz.d.ts",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**"],
        exclude = [
            "METADATA",
            "OWNERS",
            "tensorboard.google.bzl",
        ],
    ),
    tags = ["notsan"],
)
