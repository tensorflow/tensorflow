# A minimalistic profiler sampling pseudo-stacks

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

config_setting(
    name = "ruy_profiler",
    define_values = {"ruy_profiler": "true"},
)

cc_library(
    name = "instrumentation",
    srcs = ["instrumentation.cc"],
    hdrs = ["instrumentation.h"],
    defines = select({
        ":ruy_profiler": ["RUY_PROFILER"],
        "//conditions:default": [],
    }),
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
    deps = [
        ":profiler",
        ":test_instrumented_library",
        "@com_google_googletest//:gtest",
    ],
)
