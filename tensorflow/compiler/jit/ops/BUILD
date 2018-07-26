licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//tensorflow/compiler/tf2xla:internal"],
)

cc_library(
    name = "xla_ops",
    srcs = ["xla_ops.cc"],
    copts= ["-g", "-O0"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

cc_library(
    name = "parallel_check_op",
    srcs = ["parallel_check_op.cc"],
    copts= ["-g", "-O0"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)
