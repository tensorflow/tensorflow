# Description:
#   LMDB is the Lightning Memory-mapped Database.

licenses(["notice"])  # OpenLDAP Public License

exports_files(["LICENSE"])

cc_library(
    name = "lmdb",
    srcs = [
        "mdb.c",
        "midl.c",
    ],
    hdrs = [
        "lmdb.h",
        "midl.h",
    ],
    copts = [
        "-Wbad-function-cast",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-parameter",
        "-Wuninitialized",
    ],
    linkopts = [
        "-w",
        "-lpthread",
    ],
    visibility = ["//visibility:public"],
)
