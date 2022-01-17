load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hash",
    linkopts = ["-labsl_hash"],
    deps = [
        ":city",
        ":wyhash",
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
    name = "wyhash",
    linkopts = ["-labsl_wyhash"],
    visibility = ["//visibility:private"],
    deps = [
        "//absl/base:endian",
        "//absl/numeric:int128",
    ],
)
