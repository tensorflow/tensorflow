load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "program_name",
    linkopts = ["-labsl_flags_program_name"],
    visibility = [
        "//absl/flags:__pkg__",
    ],
    deps = [
        "//absl/strings",
        "//absl/synchronization",
    ],
)

cc_library(
    name = "config",
    linkopts = ["-labsl_flags_config"],
    deps = [
        ":program_name",
        "//absl/strings",
        "//absl/synchronization",
    ],
)

cc_library(
    name = "marshalling",
    linkopts = ["-labsl_flags_marshalling"],
    deps = [
        "//absl/base:log_severity",
        "//absl/strings",
        "//absl/strings:str_format",
    ],
)

cc_library(
    name = "commandlineflag_internal",
    linkopts = ["-labsl_flags_commandlineflag_internal"],
)

cc_library(
    name = "commandlineflag",
    linkopts = ["-labsl_flags_commandlineflag"],
    deps = [
        ":commandlineflag_internal",
        "//absl/strings",
        "//absl/types:optional",
    ],
)

cc_library(
    name = "private_handle_accessor",
    linkopts = ["-labsl_flags_private_handle_accessor"],
    visibility = [
        "//absl/flags:__pkg__",
    ],
    deps = [
        ":commandlineflag",
        ":commandlineflag_internal",
        "//absl/strings",
    ],
)

cc_library(
    name = "reflection",
    linkopts = ["-labsl_flags_reflection"],
    deps = [
        ":commandlineflag",
        ":commandlineflag_internal",
        ":config",
        ":private_handle_accessor",
        "//absl/container:flat_hash_map",
        "//absl/strings",
        "//absl/synchronization",
    ],
)

cc_library(
    name = "flag_internal",
    linkopts = ["-labsl_flags_internal"],
    visibility = ["//absl/base:__subpackages__"],
    deps = [
        ":commandlineflag",
        ":commandlineflag_internal",
        ":config",
        ":marshalling",
        ":reflection",
        "//absl/base",
        "//absl/memory",
        "//absl/meta:type_traits",
        "//absl/strings",
        "//absl/synchronization",
        "//absl/utility",
    ],
)

cc_library(
    name = "flag",
    linkopts = ["-labsl_flags"],
    deps = [
        ":config",
        ":flag_internal",
        ":reflection",
        "//absl/base",
        "//absl/strings",
    ],
)

cc_library(
    name = "usage_internal",
    linkopts = ["-labsl_flags_usage_internal"],
    visibility = [
        "//absl/flags:__pkg__",
    ],
    deps = [
        ":commandlineflag",
        ":config",
        ":flag",
        ":flag_internal",
        ":private_handle_accessor",
        ":program_name",
        ":reflection",
        "//absl/strings",
    ],
)

cc_library(
    name = "usage",
    linkopts = ["-labsl_flags_usage"],
    deps = [
        ":usage_internal",
        "//absl/strings",
        "//absl/synchronization",
    ],
)

cc_library(
    name = "parse",
    linkopts = ["-labsl_flags_parse"],
    deps = [
        ":commandlineflag",
        ":commandlineflag_internal",
        ":config",
        ":flag",
        ":flag_internal",
        ":private_handle_accessor",
        ":program_name",
        ":reflection",
        ":usage",
        ":usage_internal",
        "//absl/strings",
        "//absl/synchronization",
    ],
)
