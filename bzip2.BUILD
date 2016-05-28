package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD derivative

prefix_dir = "bzip2-1.0.6"

BZ2LIB_SRCS = [
    # these are in the same order as their corresponding .o files are in OBJS in
    # Makefile (rather than lexicographic order) for easy comparison (that they
    # are identical).
    "blocksort.c",
    "huffman.c",
    "crctable.c",
    "randtable.c",
    "compress.c",
    "decompress.c",
    "bzlib.c",
]

cc_library(
    name = "bz2lib",
    srcs = [prefix_dir + "/" + source for source in BZ2LIB_SRCS] +
        [prefix_dir + "/bzlib_private.h"],
    hdrs = [prefix_dir + "/bzlib.h"],
    includes = [prefix_dir],
)

cc_binary(
    name = "bzip2",
    srcs = [
        "bzip2.c",
    ],
    deps = [
        ":bz2lib",
    ],
)
