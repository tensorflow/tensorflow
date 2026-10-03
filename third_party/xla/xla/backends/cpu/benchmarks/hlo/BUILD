# HLO files for end-to-end benchmarks.

load(
    "//xla/backends/cpu/benchmarks:shared_defs.bzl",
    "CPU_BENCHMARKS_VISIBILITY",
)
load(
    ":hlo_files.bzl",
    "MEMORY_INTENSIVE_HLO_FILES",
    "REGULAR_HLO_FILES",
    "SLOW_HLO_FILES",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = CPU_BENCHMARKS_VISIBILITY,
    licenses = ["notice"],
)

LOCAL_HLO_FILES = REGULAR_HLO_FILES + SLOW_HLO_FILES + MEMORY_INTENSIVE_HLO_FILES

exports_files(LOCAL_HLO_FILES + ["hlo_files.bzl"])

filegroup(
    name = "hlo_files",
    srcs = LOCAL_HLO_FILES,
)

filegroup(
    name = "regular_hlo_files",
    srcs = REGULAR_HLO_FILES,
)

filegroup(
    name = "slow_hlo_files",
    srcs = SLOW_HLO_FILES,
)

filegroup(
    name = "memory_intensive_hlo_files",
    srcs = MEMORY_INTENSIVE_HLO_FILES,
)
