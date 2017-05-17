# Description:
#   LMDB is the Lightning Memory-mapped Database.

licenses(["notice"])  # OpenLDAP Public License

cc_library(
    name = "lmdb",
    srcs = [
        "libraries/liblmdb/mdb.c",
        "libraries/liblmdb/midl.c",
    ],
    hdrs = [
        "libraries/liblmdb/lmdb.h",
        "libraries/liblmdb/midl.h",
    ],
    includes = ["libraries/liblmdb/"],
    copts = [
        "-Wbad-function-cast",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-parameter",
        "-Wuninitialized",
    ],
    linkopts = [
        "-lpthread",
    ],
    visibility = ["//visibility:public"],
)
