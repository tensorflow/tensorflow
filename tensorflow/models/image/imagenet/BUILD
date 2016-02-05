# Description:
# Example TensorFlow models for ImageNet.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "classify_image",
    srcs = [
        "classify_image.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
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
    visibility = ["//visibility:public"],
)
