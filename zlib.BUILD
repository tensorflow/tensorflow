package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license (for zlib)

prefix_dir = "zlib-1.2.8"

cc_library(
    name = "zlib",
    srcs = glob([prefix_dir + "/*.c"]),
    hdrs = glob([prefix_dir + "/*.h"]),
    includes = [prefix_dir],
)
