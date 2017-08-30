# Description:
#   Sqlite3 library. Provides utilities for interacting
#   with sqlite3 databases.

licenses(["notice"])  # BSD/MIT-like license

exports_files(["LICENSE"])

cc_library(
    name = "sqlite",
    srcs = ["sqlite3.c"],
    hdrs = ["sqlite3.h"],
    includes = ["."],
    linkopts = ["-lm"],
    visibility = ["//visibility:public"],
)
