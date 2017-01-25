package(default_visibility = ["//visibility:public"])

config_setting(
  name = "using_poplar",
  values = {"define": "using_poplar=true"},
)

cc_library(
  name = "poplar",
)

