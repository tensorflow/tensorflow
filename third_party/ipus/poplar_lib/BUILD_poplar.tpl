# Template for the BUILD file for the generated poplar repository

package(default_visibility = ["//visibility:public"])

config_setting(
  name = "using_poplar",
  values = {"define": "using_poplar=true"},
)

cc_library(
  name = "poplar_libs",
  srcs = select({
    "@local_config_poplar//poplar:using_poplar": glob(["lib/libpoplar.*"]),
    "//conditions:default": []
  }),
)

cc_library(
  name = "poplar_headers",
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
  name = "poplar",
  deps = [":poplar_libs", ":poplar_headers"]
)



