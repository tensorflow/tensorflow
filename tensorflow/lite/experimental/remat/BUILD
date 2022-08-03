cc_library(
    name = "metadata_util",
    srcs = ["metadata_util.cc"],
    hdrs = ["metadata_util.h"],
    deps = [
        "//base",
        "//tensorflow/lite:graph_info",
    ],
)

cc_test(
    name = "metadata_util_test",
    size = "small",
    srcs = ["metadata_util_test.cc"],
    deps = [
        ":metadata_util",
        "@com_google_googletest//:gtest_main",
    ],
)
