licenses(["restricted"])  # NVIDIA proprietary license

filegroup(
    name = "nvprune",
    srcs = [
        "bin/nvprune",
    ],
    visibility = ["//visibility:public"],
)
