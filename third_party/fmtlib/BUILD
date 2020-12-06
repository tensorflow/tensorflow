package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.rst"])

cc_library(
    name = "fmtlib",
    hdrs = [
        "include/fmt/format.h",
        "include/fmt/format-inl.h",
        "include/fmt/core.h",
    ],
    srcs = [
        "src/format.cc",
    ],
    strip_include_prefix = "include/",
)
