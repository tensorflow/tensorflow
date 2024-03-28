load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hash",
    linkopts = ["-labsl_hash"],
    deps = [
        ":city",
        ":low_level_hash",
        "//absl/base:endian",
        "//absl/container:fixed_array",
        "//absl/numeric:int128",
        "//absl/strings",
        "//absl/types:optional",
        "//absl/types:variant",
        "//absl/utility",
    ],
)

cc_library(
    name = "city",
    linkopts = ["-labsl_city"],
    deps = [
        "//absl/base:endian",
    ],
)

cc_library(
    name = "low_level_hash",
    linkopts = ["-labsl_low_level_hash"],
    visibility = ["//visibility:private"],
    deps = [
        "//absl/base:endian",
        "//absl/numeric:int128",
    ],
)
