# Template for popnn

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "popnn_libs",
  srcs = select({
    "@local_config_poplar//poplar:using_poplar": glob(["**/libpopnn.*"]),
    "//conditions:default": []
  }),
)

cc_library(
  name = "popnn_headers",
  hdrs = select({
    "@local_config_poplar//poplar:using_poplar": glob(["**/*.h"]),
    "//conditions:default": []
  }),
  includes = select({
    "@local_config_poplar//poplar:using_poplar": ["include"],
    "//conditions:default": []
  }),
)

cc_library(
  name = "popnn",
  deps = [":popnn_libs", ":popnn_headers"]
)

