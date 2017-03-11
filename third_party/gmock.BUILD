# Description:
#   Google C++ Mocking Framework, a library for creating and using C++
#   mock classes.

licenses(["notice"])  # 3-clause BSD

exports_files(["LICENSE"])

cc_library(
    name = "gtest",
    srcs = [
<<<<<<< HEAD
        "googletest/src/gtest-all.cc",
        "googlemock/src/gmock-all.cc",
=======
        "googlemock/src/gmock-all.cc",
        "googletest/src/gtest-all.cc",
>>>>>>> eb8bb9e461f669f299aa031634530995bc43f92b
    ],
    hdrs = glob([
        "**/*.h",
        "googletest/src/*.cc",
        "googlemock/src/*.cc",
    ]),
    includes = [
<<<<<<< HEAD
        "googletest",
        "googletest/include",
        "googlemock",
        "googlemock/include",
=======
        "googlemock",
        "googlemock/include",
        "googletest",
        "googletest/include",
>>>>>>> eb8bb9e461f669f299aa031634530995bc43f92b
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest_main",
    srcs = ["googlemock/src/gmock_main.cc"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [":gtest"],
)
