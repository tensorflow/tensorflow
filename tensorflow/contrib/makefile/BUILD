# Necessary build rules for makefile build in our CI.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = ["**/OWNERS"],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
