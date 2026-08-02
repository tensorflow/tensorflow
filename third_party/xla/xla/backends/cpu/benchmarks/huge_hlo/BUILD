# HLO files for end-to-end benchmarks (huge).

load("//xla/backends/cpu/benchmarks:shared_defs.bzl", "CPU_BENCHMARKS_VISIBILITY")
load(
    ":hlo_files.bzl",
    "HUGE_HLO_FILES",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = CPU_BENCHMARKS_VISIBILITY,
    licenses = ["notice"],
)

exports_files(HUGE_HLO_FILES + ["hlo_files.bzl"])

filegroup(
    name = "hlo_files",
    srcs = HUGE_HLO_FILES,
)
