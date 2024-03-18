licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_import(
    name = "nccl",
    shared_library = "lib/libnccl.so.2",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "nccl_lib",
    srcs = [
        "lib/libnccl.so.2",
    ],
)
