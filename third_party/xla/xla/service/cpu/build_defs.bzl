"""build_defs for service/cpu."""

load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_SERVICE_CPU_BUILD_DEFS_USERS",
)
load("//xla/tsl:tsl.bzl", "clean_dep")

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_SERVICE_CPU_BUILD_DEFS_USERS)

def runtime_copts():
    """Returns copts used for CPU runtime libraries."""
    return (["-DEIGEN_AVOID_STL_ARRAY"] + select({
        clean_dep("//xla/tsl:android_arm"): ["-mfpu=neon"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//xla/tsl:android"): ["-O2"],
        "//conditions:default": [],
    }))
