# Description: rocPRIM library which is a set of primitives for GPU programming on AMD ROCm stack.

licenses(["notice"])  # BSD

<<<<<<< HEAD
exports_files(["LICENSE.TXT"])

load("@local_config_rocm//rocm:build_defs.bzl", "rocm_default_copts", "if_rocm")
=======
exports_files(["LICENSE.txt"])

load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm", "rocm_default_copts")
>>>>>>> upstream/master

filegroup(
    name = "rocprim_headers",
    srcs = glob([
        "hipcub/include/**",
        "rocprim/include/**",
    ]),
)

cc_library(
    name = "rocprim",
<<<<<<< HEAD
    hdrs = if_rocm([":rocprim_headers"]),
    srcs= ["rocprim_version.hpp", "hipcub_version.hpp"],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
    ],
    includes = ["hipcub/include",
             "rocprim/include",
             "rocprim/include/rocprim",
             ".",],
    visibility = ["//visibility:public"],
=======
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
>>>>>>> upstream/master
)

genrule(
    name = "rocprim_version_hpp",
<<<<<<< HEAD
    message = "Creating rocPRIM version header...",
=======
>>>>>>> upstream/master
    srcs = ["rocprim/include/rocprim/rocprim_version.hpp.in"],
    outs = ["rocprim_version.hpp"],
    cmd = ("sed " +
           "-e 's/@rocprim_VERSION_MAJOR@/1/g' " +
           "-e 's/@rocprim_VERSION_MINOR@/0/g' " +
           "-e 's/@rocprim_VERSION_PATCH@/0/g' " +
           "$< >$@"),
<<<<<<< HEAD
=======
    message = "Creating rocPRIM version header...",
>>>>>>> upstream/master
)

genrule(
    name = "hipcub_version_hpp",
<<<<<<< HEAD
    message = "Creating hipcub version header...",
=======
>>>>>>> upstream/master
    srcs = ["hipcub/include/hipcub/hipcub_version.hpp.in"],
    outs = ["hipcub_version.hpp"],
    cmd = ("sed " +
           "-e 's/@rocprim_VERSION_MAJOR@/0/g' " +
           "-e 's/@rocprim_VERSION_MINOR@/3/g' " +
           "-e 's/@rocprim_VERSION_PATCH@/0/g' " +
           "$< >$@"),
<<<<<<< HEAD
=======
    message = "Creating hipcub version header...",
>>>>>>> upstream/master
)
