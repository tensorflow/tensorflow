# Description:
#   This is a old build rule for Scikit Flow (moved to learn).

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "skflow",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/setuptools",
        "//third_party/py/sklearn",
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
