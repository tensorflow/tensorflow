licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

filegroup(
    name = "include",
    srcs = glob([
        "include/**",
    ]),
)

filegroup(
    name = "bin",
    srcs = glob([
        "bin/**",
    ]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "nvprune",
    srcs = [
        "bin/nvprune",
    ],
    visibility = ["//visibility:public"],
)
