# Template for popnn

package(default_visibility = ["//visibility:public"])

filegroup(
  name = "popnn_lib",
  srcs = glob(["lib/libpopnn.*"]),
)

cc_library(
  name = "popnn_headers",
  hdrs = glob(["**/*.h"]),
  includes = ["include"],
)
