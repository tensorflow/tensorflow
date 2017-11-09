licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "tfprof",
    srcs = [
        "__init__.py",
        "model_analyzer.py",
        "tfprof_logger.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/profiler:model_analyzer",
        "//tensorflow/python/profiler:profile_context",
        "//tensorflow/python/profiler:tfprof_logger",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
