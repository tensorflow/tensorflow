load("@local_config_sycl//sycl:build_defs.bzl", "sycl_library")

sycl_library(
    name = "oneccl",
    srcs = glob([
        "src/api.cpp",
        "src/debug.cpp",
    ]),
    hdrs = glob([
        "include/**/*.h",
        "src/**/*.h",
    ]),
    includes = [
        "include",
        "src",
    ],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        "@oneccl_v1",
    ],
)

sycl_library(
    name = "ccl_legacy",
    srcs = glob([
        "plugins/legacy/ccl_legacy.cpp",
    ]),
    hdrs = glob([
        "src/**/*.hpp",
    ]),
    includes = [
        "include",
        "src",
    ],
    deps = [
        ":oneccl",
        "@oneccl_v1",
        "@oneccl_v1//:mpi",
    ],
)

sycl_library(
    name = "libs",
    visibility = ["//visibility:public"],
    deps = [
        ":ccl_legacy",
        ":oneccl",
    ],
)
