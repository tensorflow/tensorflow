# Description:
# Profiler test data packages.

load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "profiler_testdata",
    srcs = [
        "ckpt.data-00000-of-00001",
        "ckpt.index",
        "ckpt.meta",
        "graph.pbtxt",
        "run_meta",
        "tfprof_log",
    ],
    visibility = ["//tensorflow/core/profiler/internal:__pkg__"],
)
