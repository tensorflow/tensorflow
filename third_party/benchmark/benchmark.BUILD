package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache

exports_files(["LICENSE"])

config_setting(
    name = "qnx",
    constraint_values = ["@platforms//os:qnx"],
    values = {
        "cpu": "x64_qnx",
    },
    visibility = [":__subpackages__"],
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
    values = {
        "cpu": "x64_windows",
    },
    visibility = [":__subpackages__"],
)

config_setting(
    name = "macos",
    constraint_values = [
        "@platforms//os:macos",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "benchmark",
    srcs = glob(
        [
            "src/*.cc",
            "src/*.h",
        ],
        exclude = ["src/benchmark_main.cc"],
    ),
    hdrs = ["include/benchmark/benchmark.h"],
    linkopts = select({
        ":windows": ["-DEFAULTLIB:shlwapi.lib"],
        ":macos": ["-lpthread"],
        "//conditions:default": [
            "-pthread",
            "-lrt",
        ],
    }),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "benchmark_main",
    srcs = ["src/benchmark_main.cc"],
    hdrs = ["include/benchmark/benchmark.h"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [":benchmark"],
)

cc_library(
    name = "benchmark_internal_headers",
    hdrs = glob(["src/*.h"]),
)
