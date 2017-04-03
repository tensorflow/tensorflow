# Template for popnn

package(default_visibility = ["//visibility:public"])

filegroup(
  name = "popnn_lib",
  srcs = glob([
    "lib/libenigma.*",
    "lib/libpopnn.*",
    "lib/libpoplin.*",
    "lib/libpopconv.*",
    "lib/libpopreduce.*",
    "lib/libpopstd.*",
  ]),
)

cc_library(
  name = "popnn_headers",
  hdrs = glob(["**/*.h"]),
  includes = ["include"],
)
