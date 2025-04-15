# Description:
#   sqlite3 is a serverless SQL RDBMS.

licenses(["unencumbered"])  # Public Domain

SQLITE_COPTS = [
    "-DSQLITE_ENABLE_JSON1",
    "-DHAVE_DECL_STRERROR_R=1",
    "-DHAVE_STDINT_H=1",
    "-DHAVE_INTTYPES_H=1",
    "-D_FILE_OFFSET_BITS=64",
    "-D_REENTRANT=1",
] + select({
    "@local_xla//xla/tsl:windows": [
        "-DSQLITE_MAX_TRIGGER_DEPTH=100",
    ],
    "@local_xla//xla/tsl:macos": [
        "-Os",
        "-DHAVE_GMTIME_R=1",
        "-DHAVE_LOCALTIME_R=1",
        "-DHAVE_USLEEP=1",
    ],
    "//conditions:default": [
        "-Os",
        "-DHAVE_FDATASYNC=1",
        "-DHAVE_GMTIME_R=1",
        "-DHAVE_LOCALTIME_R=1",
        "-DHAVE_POSIX_FALLOCATE=1",
        "-DHAVE_USLEEP=1",
    ],
})

# Production build of SQLite library that's baked into TensorFlow.
cc_library(
    name = "org_sqlite",
    srcs = ["sqlite3.c"],
    hdrs = [
        "sqlite3.h",
        "sqlite3ext.h",
    ],
    copts = SQLITE_COPTS,
    defines = [
        # This gets rid of the bloat of deprecated functionality. It
        # needs to be listed here instead of copts because it's actually
        # referenced in the sqlite3.h file.
        "SQLITE_OMIT_DEPRECATED",
    ],
    linkopts = select({
        "@local_xla//xla/tsl:windows": [],
        "//conditions:default": [
            "-ldl",
            "-lpthread",
        ],
    }),
    visibility = ["//visibility:public"],
)

# This is a Copybara sync helper for Google.
py_library(
    name = "python",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)
