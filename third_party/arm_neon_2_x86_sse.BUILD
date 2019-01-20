# Description:
#   NEON2SSE - a header file redefining ARM Neon intrinsics in terms of SSE intrinsics
#              allowing neon code to compile and run on x64/x86 workstantions.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # 3-Clause BSD

exports_files([
    "LICENSE",
])

cc_library(
    name = "arm_neon_2_x86_sse",
    hdrs = ["NEON_2_SSE.h"],
)
