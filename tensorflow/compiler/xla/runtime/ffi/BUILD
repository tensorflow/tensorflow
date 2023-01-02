load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_cloud")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "ffi_abi",
    hdrs = ["ffi_abi.h"],
    compatible_with = get_compatible_with_cloud(),
)

cc_library(
    name = "ffi_api",
    hdrs = ["ffi_api.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":ffi_abi",
        ":ffi_c_api_hdrs",
    ],
)

cc_library(
    name = "ffi_c_api_hdrs",
    hdrs = ["ffi_c_api.h"],
    compatible_with = get_compatible_with_cloud(),
)
