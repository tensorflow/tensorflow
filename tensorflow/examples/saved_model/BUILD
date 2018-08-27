# Description: SavedModel half plus two example.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "saved_model_half_plus_two",
    srcs = [
        "saved_model_half_plus_two.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:lib",
        "//tensorflow/python/saved_model:main_op",
    ],
)
