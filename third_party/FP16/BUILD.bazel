# Description:
#   C/C++ library for conversion to/from half-precision floating-point formats

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "FP16",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    strip_include_prefix = "include",
)
