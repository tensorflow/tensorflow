# TensorFlow API backwards compatibility test goldens.

package(
    default_visibility = ["//tensorflow/tools/api:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "api_golden",
    srcs = glob(["*.pbtxt"]),
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
