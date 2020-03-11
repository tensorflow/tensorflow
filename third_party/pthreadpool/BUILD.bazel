# Description:
#   Portable pthread-based thread pool for C and C++

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "pthreadpool",
    srcs = [
        "src/threadpool-pthreads.c",
        "src/threadpool-utils.h",
    ],
    hdrs = [
        "include/pthreadpool.h",
    ],
    copts = [
        "-O2",
    ],
    defines = [
        "PTHREADPOOL_NO_DEPRECATED_API",
    ],
    includes = [
        "include",
    ],
    strip_include_prefix = "include",
    deps = [
        "@FXdiv",
    ],
)
