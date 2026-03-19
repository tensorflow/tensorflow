load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "status",
    linkopts = ["-labsl_status"],
    deps = [
        "//absl/base:atomic_hook",
        "//absl/base:raw_logging_internal",
        "//absl/container:inlined_vector",
        "//absl/debugging:stacktrace",
        "//absl/debugging:symbolize",
        "//absl/strings",
        "//absl/strings:cord",
        "//absl/strings:str_format",
        "//absl/types:optional",
    ],
)

cc_library(
    name = "statusor",
    linkopts = ["-labsl_statusor"],
    deps = [
        ":status",
        "//absl/base:raw_logging_internal",
        "//absl/strings",
        "//absl/types:variant",
        "//absl/utility",
    ],
)
