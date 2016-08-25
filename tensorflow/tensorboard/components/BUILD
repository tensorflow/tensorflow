package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        [
            "tf-*/**/*",
            "vz-*/**/*",
        ],
        exclude = [
            "**/tf_model_zoo/*",
            "**/METADATA",
            "**/OWNERS",
        ],
    ) + [
        "BUILD",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)
