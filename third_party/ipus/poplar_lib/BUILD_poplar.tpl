# Template for the BUILD file for the generated poplar repository

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "popc",
    srcs = ["bin/popc"],
)

filegroup(
  name = "poplar_lib",
  srcs = glob([
    "lib/libpoplar.*",
    "lib/libxgraph.*"
  ]),
)

cc_library(
  name = "poplar_headers",
  hdrs = glob(["**/*.h"]),
  includes = ["include"],
)
