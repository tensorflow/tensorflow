# The rapids-logger project defines an easy way to produce a project-specific logger using the excellent spdlog package
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "rapids_logger",
    srcs = ["src/logger.cpp"],
    hdrs = glob(["include/rapids_logger/*.h*"]),
    copts = [
        "-std=c++17",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@spdlog",
    ],
)

cc_test(
    name = "smoke_test",
    srcs = ["smoke_test.cc"],
    deps = [
        ":rapids_logger",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
