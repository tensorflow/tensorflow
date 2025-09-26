# fmt is an open-source formatting library providing a fast and safe alternative to C stdio and C++ iostreams.
licenses(["notice"])  # MIT

exports_files(["LICENSE"])

cc_library(
    name = "fmt",
    hdrs = glob(["include/fmt/*.h"]),
    copts = ["-fexceptions"],
    defines = [
        "FMT_HEADER_ONLY=1",
        "FMT_USE_USER_DEFINED_LITERALS=0",
    ],
    features = ["-use_header_modules"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "fmt_smoke_test",
    srcs = [
        "test/assert-test.cc",
        "test/header-only-test.cc",
        "test/test-main.cc",
    ],
    deps = [
        ":fmt",
        "@com_google_googletest//:gtest",
    ],
)
