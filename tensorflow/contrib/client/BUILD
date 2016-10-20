# Description:
# Tensorflow Client utils and frameworks

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])


py_library(
    name = "client_py",
    srcs = [
        "__init__.py",
        "python/framework/__init__.py",
        "python/framework/session_run.py",
    ],
    srcs_version = "PY2AND3",
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
