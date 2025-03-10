load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "time",
    linkopts = [
        "-labsl_time",
        "-labsl_civil_time",
        "-labsl_time_zone",
    ],
    deps = [
        "//absl/base",
        "//absl/base:raw_logging_internal",
        "//absl/numeric:int128",
        "//absl/strings",
    ],
)
