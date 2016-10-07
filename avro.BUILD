package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "avrocpp",
    srcs = glob(
        [
            "impl/**/*.cc",
            "impl/**/*.hh",
        ],
        exclude = [
            "impl/avrogencpp.cc",
        ],
    ),
    hdrs = glob(["api/**/*.hh"]),
    includes = ["api"],
    deps = [
        "@boost_archive//:boost",
        "@boost_archive//:filesystem",
        "@boost_archive//:iostreams",
        "@boost_archive//:system",
    ],
)

cc_binary(
    name = "avrogencpp",
    srcs = ["impl/avrogencpp.cc"],
    deps = [
        ":avrocpp",
        "@boost_archive//:program_options",
    ],
)
