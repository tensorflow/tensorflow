# Template for the BUILD file for the generated poplar repository

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "poplar_libs",
  srcs = glob(["lib/libpoplar.*"]),
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



