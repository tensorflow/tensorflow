load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Internal data structure for efficiently detecting mutex dependency cycles
cc_library(
    name = "graphcycles_internal",
    linkopts = ["-labsl_graphcycles_internal"],
    visibility = [
        "//absl:__subpackages__",
    ],
    deps = [
        "//absl/base",
        "//absl/base:malloc_internal",
        "//absl/base:raw_logging_internal",
    ],
)

cc_library(
    name = "synchronization",
    linkopts = [
        "-labsl_synchronization",
        "-pthread",
    ],
    deps = [
        ":graphcycles_internal",
        "//absl/base",
        "//absl/base:atomic_hook",
        "//absl/base:dynamic_annotations",
        "//absl/base:malloc_internal",
        "//absl/base:raw_logging_internal",
        "//absl/debugging:stacktrace",
        "//absl/debugging:symbolize",
        "//absl/time",
    ],
)
