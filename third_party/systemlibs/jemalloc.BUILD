licenses(["notice"])  # BSD

filegroup(
    name = "COPYING",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "jemalloc_headers",
    defines = [
        "jemalloc_posix_memalign=posix_memalign",
        "jemalloc_malloc=malloc",
        "jemalloc_realloc=realloc",
        "jemalloc_free=free",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "jemalloc_impl",
    linkopts = ["-ljemalloc"],
    defines = [
        "jemalloc_posix_memalign=posix_memalign",
        "jemalloc_malloc=malloc",
        "jemalloc_realloc=realloc",
        "jemalloc_free=free",
    ],
    visibility = ["//visibility:public"],
    deps = [":jemalloc_headers"],
)
