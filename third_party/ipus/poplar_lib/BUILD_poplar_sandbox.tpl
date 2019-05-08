# Template for the BUILD file for the generated poplar repository

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "popc",
    srcs = ["poplar/bin/popc"],
)

filegroup(
  name = "poplar_lib",
  srcs = glob([
    "poplar/lib/libpoplar.*",
    "poplibs/lib/libpoplin.*",
    "poplibs/lib/libpopnn.*",
    "poplibs/lib/libpopops.*",
    "poplibs/lib/libpoprand.*",
    "poplibs/lib/libpopsys.*",
    "poplibs/lib/libpoputil.*",
    "poplibs/lib/libpoplibs_support.*",
  ]),
)

cc_library(
  name = "poplar_headers",
  hdrs = glob(["**/*.h"]),
  includes = ["poplar/include", "poplibs/include"],
)
