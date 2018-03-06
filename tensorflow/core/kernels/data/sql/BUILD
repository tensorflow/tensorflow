# Description:
#   SQL library.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        include = ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "sql",
    srcs = [
        "driver_manager.cc",
        "sqlite_query_connection.cc",
    ],
    hdrs = [
        "driver_manager.h",
        "query_connection.h",
        "sqlite_query_connection.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/data:dataset",
        "//tensorflow/core/lib/db:sqlite",
    ],
)
