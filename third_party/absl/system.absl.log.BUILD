load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

[cc_library(
    name = n,
) for n in [
    "check_impl",
    "check_op",
    "conditions",
    "config",
    "core_headers",
    "log_impl",
    "log_message",
    "log_sink_set",
    "strip",
]]

cc_library(
    name = "absl_log",
    linkopts = ["-labsl_log_absl_log"],
)

cc_library(
    name = "check",
    linkopts = ["-labsl_log_check"],
)

cc_library(
    name = "die_if_null",
    linkopts = ["-labsl_log_die_if_null"],
    deps = [
        ":log",
    ],
)

cc_library(
    name = "flags",
    linkopts = ["-labsl_log_flags"],
    deps = [
        ":globals",
    ],
)

cc_library(
    name = "globals",
    linkopts = ["-labsl_log_globals"],
)

cc_library(
    name = "log",
    linkopts = ["-labsl_log_log"],
)

cc_library(
    name = "log_entry",
    linkopts = ["-labsl_log_log_entry"],
)

cc_library(
    name = "log_sink",
    linkopts = ["-labsl_log_log_sink"],
    deps = [
        "log_entry",
    ],
)

cc_library(
    name = "log_sink_registry",
    linkopts = ["-labsl_log_log_sink_registry"],
    deps = [
        "log_sink",
    ],
)

cc_library(
    name = "log_streamer",
    linkopts = ["-labsl_log_log_streamer"],
    deps = [
        "absl_log",
    ],
)
