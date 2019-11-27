package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "common",
    srcs = ["common.c"],
    hdrs = [
        "builtin_op_data.h",
        "common.h",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
)

# For use with library targets that can't use relative paths.
exports_files(["common.h"])

# Test the C extension API code.
cc_test(
    name = "common_test",
    size = "small",
    srcs = ["common_test.cc"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "builtin_op_data_test",
    size = "small",
    srcs = ["builtin_op_data_test.cc"],
    copts = ["-Wno-unused-variable"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest",
    ],
)
