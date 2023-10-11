load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "ffi_hdrs",
    srcs = [
        "ffi_abi.h",
        "ffi_api.h",
        "ffi_c_api.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ffi_abi",
    hdrs = ["ffi_abi.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ffi_api",
    hdrs = ["ffi_api.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":ffi_abi",
        ":ffi_c_api_hdrs",
    ],
)

cc_library(
    name = "ffi_c_api_hdrs",
    hdrs = ["ffi_c_api.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)
