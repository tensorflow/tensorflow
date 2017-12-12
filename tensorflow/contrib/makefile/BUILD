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

sh_test(
    name = "build_all_linux",
    size = "enormous",
    srcs = ["build_all_linux.sh"],
    data = [
        "//tensorflow:all_opensource_files",
        "//third_party/eigen3:all_files",
        "//third_party/fft2d:all_files",
    ],
    tags = [
        "manual",
        "no_gpu",
        "no_oss",
        "notap",
    ],
)
