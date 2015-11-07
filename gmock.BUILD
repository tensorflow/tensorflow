cc_library(
    name = "gtest",
    srcs = [
        "gmock-1.7.0/gtest/src/gtest-all.cc",
        "gmock-1.7.0/src/gmock-all.cc",
    ],
    includes = [
        "gmock-1.7.0",
        "gmock-1.7.0/gtest",
        "gmock-1.7.0/gtest/include",
        "gmock-1.7.0/include",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest_main",
    srcs = ["gmock-1.7.0/src/gmock_main.cc"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [":gtest"],
)
