# Description:
#   Libraries for storing tensors in SQL databases.

load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_copts")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "sqlite",
    srcs = ["sqlite.cc"],
    hdrs = ["sqlite.h"],
    copts = tf_copts(),
    deps = [
        ":snapfn",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@org_sqlite",
    ],
)

cc_library(
    name = "snapfn",
    srcs = ["snapfn.cc"],
    copts = tf_copts() + ["-DSQLITE_OMIT_LOAD_EXTENSION"],
    linkstatic = 1,
    deps = [
        "@org_sqlite",
        "@snappy",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "sqlite_test",
    size = "small",
    srcs = ["sqlite_test.cc"],
    deps = [
        ":sqlite",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
