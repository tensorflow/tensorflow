package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD derivative

cc_library(
    name = "bz2lib",
    srcs = [
        # These are in the same order as their corresponding .o files are in
        # OBJS in Makefile (rather than lexicographic order) for easy
        # comparison (that they are identical.)
        "blocksort.c",
        "huffman.c",
        "crctable.c",
        "randtable.c",
        "compress.c",
        "decompress.c",
        "bzlib.c",
        "bzlib_private.h",
    ],
    hdrs = ["bzlib.h"],
    includes = ["."],
)

cc_binary(
    name = "bzip2",
    srcs = ["bzip2.c"],
    deps = [":bz2lib"],
)
