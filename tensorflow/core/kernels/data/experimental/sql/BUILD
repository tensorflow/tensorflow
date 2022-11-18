# Description:
#   SQL library.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
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
        "//tensorflow/core/lib/db:sqlite",
    ],
)
