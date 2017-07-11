package(default_visibility = ["//visibility:public"])

licenses(["notice"]) # OpenIB.org BSD license (MIT variant)

exports_files(["COPYING"])

cc_library(
    name = "ibverbs",
    hdrs = [
        "include/infiniband/sa.h",
        "include/infiniband/verbs.h",
    ],
    defines = [
        "NRESOLVE_NEIGH=1",
        "STREAM_CLOEXEC=\\\"e\\\"",
        "IBV_CONFIG_DIR=\\\"/etc/libibverbs.d\\\"",
    ],
    srcs = [
        "src/cmd.c",
        "src/compat-1_0.c",
        "src/device.c",
        "src/enum_strs.c",
        "src/ibverbs.h",
        "src/init.c",
        "src/marshall.c",
        "src/memory.c",
        "src/sysfs.c",
        "src/verbs.c",
    ],
    includes = ["include"],
    include_prefix = "infiniband",
)
