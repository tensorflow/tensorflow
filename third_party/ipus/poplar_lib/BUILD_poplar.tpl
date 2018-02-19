# Template for the BUILD file for the generated poplar repository

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "popc",
    srcs = ["bin/popc"],
)

filegroup(
  name = "poplar_lib",
  srcs = glob([
    "lib/libpopconv.*",
    "lib/libpoplar.*",
    "lib/libpoplin.*",
    "lib/libpopnn.*",
    "lib/libpopops.*",
    "lib/libpoprand.*",
    "lib/libpoputil.*",
  ]),
)

cc_library(
  name = "poplar_headers",
  hdrs = glob(["**/*.h"]),
  includes = ["include"],
)
