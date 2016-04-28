# Description:
#   TensorFlow Studio - UI for TensorFlow.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_binary(
    name = "studio",
    srcs = ["studio.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/learn",
        "//tensorflow/contrib/studio/core:server",
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
