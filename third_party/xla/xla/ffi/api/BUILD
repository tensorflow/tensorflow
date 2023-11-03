load("@local_tsl//tsl:tsl.default.bzl", "filegroup")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "api_headers",
    srcs = ["api.h"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "c_api_headers",
    srcs = ["c_api.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "api",
    hdrs = [":api_headers"],
    visibility = ["//visibility:public"],
    deps = [":c_api"],
)

cc_library(
    name = "c_api",
    hdrs = ["c_api.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "c_api_internal",
    hdrs = ["c_api_internal.h"],
    visibility = ["//visibility:public"],
    deps = [":c_api"],
)

cc_library(
    name = "ffi",
    hdrs = ["ffi.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":api",
        ":c_api",
    ],
)
