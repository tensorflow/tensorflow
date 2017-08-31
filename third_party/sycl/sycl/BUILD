# Description:
# A minimal BUILD file to make template files in this folder available. Without this BUILD file,
# bazel returns errors when trying to access tpl files in this folder.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

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
