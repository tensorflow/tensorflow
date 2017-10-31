# Description:
#   TensorBoard database code.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "schema",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

tf_cc_test(
    name = "schema_test",
    srcs = ["schema_test.cc"],
    deps = [
        ":schema",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["*"]),
    visibility = ["//tensorflow:__pkg__"],
)
