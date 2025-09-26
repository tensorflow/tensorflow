# Kokkos C++ Performance Portability Programming Ecosystem: The Programming Model - Parallel Execution and Memory Abstraction
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "mdspan",
    hdrs = glob(["tpls/mdspan/include/**/*.hpp"]),
    copts = ["-fexceptions"],
    includes = ["tpls/mdspan/include"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "smoke_test",
    srcs = ["smoke_test.cc"],
    copts = ["-fexceptions"],
    deps = [
        ":mdspan",
    ],
)
