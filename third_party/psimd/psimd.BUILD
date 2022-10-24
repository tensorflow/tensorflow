# Description:
#   Portable 128-bit SIMD intrinsics

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "psimd",
    hdrs = glob(["include/psimd.h"]),
    includes = ["include"],
    strip_include_prefix = "include",
)
