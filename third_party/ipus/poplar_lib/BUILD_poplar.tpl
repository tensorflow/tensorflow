# Template for the BUILD file for the generated poplar repository

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "popc",
    srcs = ["bin/popc"],
)

filegroup(
  name = "poplar_lib",
  srcs = glob(["lib/libpoplar.*"]),
)

cc_library(
  name = "poplar_libs",
  srcs = glob(["lib/libpoplar.*"]),
  alwayslink = 1,
)

cc_library(
  name = "poplar_headers",
  hdrs = glob(["**/*.h"]),
  includes = ["include"],
)

cc_library(
  name = "poplar",
  deps = [":poplar_libs", ":poplar_headers"]
)



