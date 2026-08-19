"""BUILD file for libdrm headers."""

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT-style license

# Export just the headers needed by ROCm
cc_library(
    name = "drm_headers",
    hdrs = glob([
        "include/drm/*.h",
    ]),
    include_prefix = "libdrm",
    strip_include_prefix = "include/drm",
    visibility = ["//visibility:public"],
)
