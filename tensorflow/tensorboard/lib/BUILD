# Description:
# BUILD rules for the static resources in TensorBoard.

package(default_visibility = [
    "//tensorflow:internal",
])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

filegroup(
    name = "all_files",
    srcs = glob(
        [
            "**/*",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
