# Description:
#   Google C++ Mocking Framework, a library for creating and using C++
#   mock classes.

licenses(["notice"])  # 3-clause BSD

exports_files(["LICENSE"])

cc_library(
    name = "gtest",
    srcs = [
        "gtest/src/gtest-all.cc",
        "src/gmock-all.cc",
    ],
    hdrs = glob([
        "**/*.h",
        "gtest/src/*.cc",
        "src/*.cc",
    ]),
    includes = [
        ".",
        "gtest",
        "gtest/include",
        "include",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest_main",
    srcs = ["src/gmock_main.cc"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [":gtest"],
)
