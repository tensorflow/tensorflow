# Description:
#   contains Scikit Flow sub-project with high level tensorflow API.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "skflow",
    srcs = [
        "__init__.py",
        "models.py",
        "trainer.py",
        "monitors.py",
    ] + glob([
        "estimators/*.py",
        "addons/*.py",
        "io/*.py",
        "ops/*.py",
        "preprocessing/*.py",
    ]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "skflow_test",
    srcs = glob(["tests/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":skflow",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ]
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

