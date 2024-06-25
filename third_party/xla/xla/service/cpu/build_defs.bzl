"""build_defs for service/cpu."""

load("//xla/tsl:tsl.bzl", "clean_dep")

def runtime_copts():
    """Returns copts used for CPU runtime libraries."""
    return (["-DEIGEN_AVOID_STL_ARRAY"] + select({
        clean_dep("//xla/tsl:android_arm"): ["-mfpu=neon"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//xla/tsl:android"): ["-O2"],
        "//conditions:default": [],
    }))
