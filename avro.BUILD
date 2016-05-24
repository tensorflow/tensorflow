package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

prefix_dir = "avro-cpp-1.8.0"

cc_library(
    name = "avrocpp",
    srcs = glob(
        [
            prefix_dir + "/impl/**/*.cc",
            prefix_dir + "/impl/**/*.hh",
        ],
        exclude = [
            prefix_dir + "/impl/avrogencpp.cc",
        ],
    ),
    hdrs = glob([prefix_dir + "/api/**/*.hh"]),
    includes = [prefix_dir + "/api"],
    deps = [
        "@boost_archive//:boost",
        "@boost_archive//:filesystem",
        "@boost_archive//:iostreams",
        "@boost_archive//:system",
    ],
)

cc_binary(
    name = "avrogencpp",
    srcs = [prefix_dir + "/impl/avrogencpp.cc"],
    deps = [
        ":avrocpp",
        "@boost_archive//:program_options",
    ],
)
