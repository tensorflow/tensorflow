licenses(["notice"])  # Apache 2.0

load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl")
load("platform", "sycl_library_path")

load("platform", "readlink_command")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE.text"])

config_setting(
    name = "using_sycl",
    values = {
        "define": "using_sycl=true",
    },
)

cc_library(
    name = "sycl_headers",
    hdrs = glob([
        "**/*.h",
    ]),
    includes = [".", "include"],
)

cc_library(
    name = "syclrt",
    srcs = [
        sycl_library_path("ComputeCpp")
    ],
    data = [
        sycl_library_path("ComputeCpp")
    ],
    includes = ["include/"],
    linkstatic = 0,
)

cc_library(
    name = "sycl",
    deps = if_sycl([
        ":sycl_headers",
        ":syclrt",
    ]),
)
