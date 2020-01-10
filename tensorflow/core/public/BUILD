package(
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(
    srcs = [
        "session.h",
        "session_options.h",
        "version.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "session.h",
        "session_options.h",
        "version.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

cc_library(
    name = "version",
    hdrs = ["version.h"],
)
