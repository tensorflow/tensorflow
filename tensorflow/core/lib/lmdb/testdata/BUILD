# Description:
# lmdb test data packages.

load("//tensorflow:tensorflow.bzl", "filegroup")

package(
    licenses = ["notice"],
)

filegroup(
    name = "lmdb_testdata",
    testonly = 1,
    srcs = [
        # A simple key-value store:
        #   0 : 'b'
        #   1 : 'b'
        #    ...
        #   9 : 'b'
        # Which is then overwritten with:
        #   0 : 'a'
        #   1 : 'b'
        #    ...
        #   9 : 'j'
        "data.mdb",
        # LMDB, being a memory-mapped database, uses a different file format on
        # big-endian systems.
        "data_bigendian.mdb",
    ],
    visibility = ["//visibility:public"],
)
