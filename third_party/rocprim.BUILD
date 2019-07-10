# Description: rocPRIM library which is a set of primitives for GPU programming on AMD ROCm stack.

licenses(["notice"])  # BSD

exports_files(["LICENSE.txt"])

load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm", "rocm_default_copts")

filegroup(
    name = "rocprim_headers",
    srcs = glob([
        "hipcub/include/**",
        "rocprim/include/**",
    ]),
)

cc_library(
    name = "rocprim",
    srcs = [
        "hipcub_version.hpp",
        "rocprim_version.hpp",
    ],
    hdrs = if_rocm([":rocprim_headers"]),
    includes = [
        ".",
        "hipcub/include",
        "rocprim/include",
        "rocprim/include/rocprim",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

genrule(
    name = "rocprim_version_hpp",
    srcs = ["rocprim/include/rocprim/rocprim_version.hpp.in"],
    outs = ["rocprim_version.hpp"],
    cmd = ("sed " +
           "-e 's/@rocprim_VERSION_MAJOR@/1/g' " +
           "-e 's/@rocprim_VERSION_MINOR@/0/g' " +
           "-e 's/@rocprim_VERSION_PATCH@/0/g' " +
           "$< >$@"),
    message = "Creating rocPRIM version header...",
)

genrule(
    name = "hipcub_version_hpp",
    srcs = ["hipcub/include/hipcub/hipcub_version.hpp.in"],
    outs = ["hipcub_version.hpp"],
    cmd = ("sed " +
           "-e 's/@rocprim_VERSION_MAJOR@/0/g' " +
           "-e 's/@rocprim_VERSION_MINOR@/3/g' " +
           "-e 's/@rocprim_VERSION_PATCH@/0/g' " +
           "$< >$@"),
    message = "Creating hipcub version header...",
)
