package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "icu4c/LICENSE",
)

filegroup(
    name = "icu4j/main/shared/licenses/LICENSE",
)

cc_library(
    name = "headers",
)

cc_library(
    name = "common",
    deps = [
        ":icuuc",
    ],
)

cc_library(
    name = "icuuc",
    linkopts = ["-licuuc"],
    visibility = ["//visibility:private"],
)
