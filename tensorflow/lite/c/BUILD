package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "c_api_internal",
    srcs = ["c_api_internal.c"],
    hdrs = [
        "builtin_op_data.h",
        "c_api_internal.h",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
)

# Test the C extension API code.
cc_test(
    name = "c_api_internal_test",
    size = "small",
    srcs = ["c_api_internal_test.cc"],
    deps = [
        ":c_api_internal",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "builtin_op_data_test",
    size = "small",
    srcs = ["builtin_op_data_test.cc"],
    copts = ["-Wno-unused-variable"],
    deps = [
        ":c_api_internal",
        "@com_google_googletest//:gtest",
    ],
)
