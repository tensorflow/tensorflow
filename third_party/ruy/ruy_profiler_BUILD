# A minimalistic profiler sampling pseudo-stacks

load("//ruy:build_defs.oss.bzl", "ruy_linkopts_thread_standard_library")

package(
    licenses = ["notice"],  # Apache 2.0
)

config_setting(
    name = "ruy_profiler",
    define_values = {"ruy_profiler": "true"},
)

# Used to build TFLite Micro RUY dependency for embedded targets outside of the
# RUY source tree.
filegroup(
    name = "ruy_instrumentation_header",
    srcs = ["instrumentation.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "instrumentation",
    srcs = ["instrumentation.cc"],
    hdrs = ["instrumentation.h"],
    defines = select({
        ":ruy_profiler": ["RUY_PROFILER"],
        "//conditions:default": [],
    }),
    linkopts = ruy_linkopts_thread_standard_library(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "profiler",
    srcs = [
        "profiler.cc",
        "treeview.cc",
    ],
    hdrs = [
        "profiler.h",
        "treeview.h",
    ],
    linkopts = ruy_linkopts_thread_standard_library(),
    visibility = ["//visibility:public"],
    deps = [":instrumentation"],
)

cc_library(
    name = "test_instrumented_library",
    testonly = True,
    srcs = ["test_instrumented_library.cc"],
    hdrs = ["test_instrumented_library.h"],
    deps = [":instrumentation"],
)

cc_test(
    name = "test",
    srcs = ["test.cc"],
    linkopts = ruy_linkopts_thread_standard_library(),
    deps = [
        ":profiler",
        ":test_instrumented_library",
        "//ruy:gtest_wrapper",
    ],
)
