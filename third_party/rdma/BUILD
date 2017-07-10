package(default_visibility = ["//visibility:public"])

licenses(["notice"]) # OpenIB.org BSD license (MIT variant)

exports_files(["COPYING.BSD_MIT"])

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

cc_library(
    name = "ibverbs",
    hdrs = [
        "verbs.h",
        "sa.h",
    ],
    include_prefix = "infiniband",
)

cc_library(
    name = "rdmacm",
    hdrs = [
        "rdma_cma.h",
        "rdma_verbs.h",
    ],
    deps = ["ibverbs"],
    include_prefix = "rdma",
)
