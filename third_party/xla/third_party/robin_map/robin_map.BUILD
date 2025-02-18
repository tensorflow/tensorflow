licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "robin_map",
    hdrs = [
        "include/tsl/robin_growth_policy.h",
        "include/tsl/robin_hash.h",
        "include/tsl/robin_map.h",
        "include/tsl/robin_set.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    includes = ["."],
    strip_include_prefix = "include",
)
