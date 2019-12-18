# Description:
# Go API for TensorFlow.

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_shared_library_deps",
)

sh_test(
    name = "test",
    size = "small",
    srcs = ["test.sh"],
    data = [
        ":all_files",  # Go sources
        "//tensorflow/c:headers",  # C library header
        "//tensorflow/c/eager:headers",  # Eager C library header
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",  # Testdata for LoadSavedModel
    ] + tf_shared_library_deps(),
    # TODO: Enable this test again once protos are supported by bazel.
    tags = ["manual"],
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
