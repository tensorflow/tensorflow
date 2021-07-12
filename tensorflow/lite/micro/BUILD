package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

# TODO(b/187728891): Remove once we not longer have a dependency from
# lite/kernels:op_macros to micro:debug_log
cc_library(
    name = "debug_log",
    srcs = [
        "debug_log.cc",
    ],
    hdrs = [
        "debug_log.h",
    ],
)
