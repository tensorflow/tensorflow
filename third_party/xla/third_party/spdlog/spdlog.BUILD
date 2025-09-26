# Fast C++ logging library. Header-only.
licenses(["notice"])  # MIT

exports_files(["LICENSE"])

cc_library(
    name = "spdlog",
    hdrs = glob(["include/spdlog/**/*.h"]),
    defines = [
        "SPDLOG_FMT_EXTERNAL",
    ],
    features = ["-parse_headers"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:node_hash_map",
        "@fmt",
    ],
)

cc_test(
    name = "smoke_test",
    srcs = [
        "smoke_test.cc",  # lightweight test file
    ],
    copts = [
        "-DSPDLOG_FMT_EXTERNAL",
        "-fexceptions",
    ],
    deps = [
        ":spdlog",
        "@fmt",
    ],
)
