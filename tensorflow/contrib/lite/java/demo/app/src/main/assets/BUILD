package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "assets_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "BUILD",
        ],
    ),
    visibility = ["//visibility:public"],
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
