# Description: Utilities.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "**/google_*",
        ],
    ),
)

cc_library(
    name = "periodic_function_dynamic",
    srcs = ["periodic_function.cc"],
    hdrs = ["periodic_function.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "periodic_function",
    visibility = ["//visibility:public"],
    deps = [
        ":periodic_function_dynamic",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "periodic_function_test",
    size = "small",
    srcs = ["periodic_function_test.cc"],
    deps = [
        ":periodic_function_dynamic",
        "//tensorflow/contrib/batching/test_util:fake_clock_env",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
