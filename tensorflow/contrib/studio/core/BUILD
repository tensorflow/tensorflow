# Description:
#   Core parts of TensorFlow Studio.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "server",
    srcs = ["server.py", "http.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/learn",
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
