# Ruy is not BLAS

# TODO(b/123403203) actually make TFLite use ruy.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "check_macros",
    hdrs = ["check_macros.h"],
    deps = ["//tensorflow/lite/kernels/internal:types"],
)

cc_library(
    name = "opt_set",
    hdrs = ["opt_set.h"],
)

cc_library(
    name = "time",
    hdrs = ["time.h"],
)

cc_library(
    name = "spec",
    hdrs = ["spec.h"],
)

cc_library(
    name = "size_util",
    hdrs = ["size_util.h"],
    deps = [":check_macros"],
)

cc_library(
    name = "tune",
    srcs = [
        "tune.cc",
    ],
    hdrs = [
        "tune.h",
    ],
    deps = [
        ":opt_set",
        ":time",
    ],
)

cc_test(
    name = "tune_test",
    srcs = ["tune_test.cc"],
    deps = [
        ":tune",
        "@com_google_googletest//:gtest",
    ],
)

cc_binary(
    name = "tune_tool",
    srcs = ["tune_tool.cc"],
    deps = [
        ":tune",
    ],
)

cc_library(
    name = "allocator",
    srcs = [
        "allocator.cc",
    ],
    hdrs = [
        "allocator.h",
    ],
    deps = [
        ":check_macros",
        ":size_util",
    ],
)

cc_test(
    name = "allocator_test",
    srcs = ["allocator_test.cc"],
    deps = [
        ":allocator",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "block_map",
    srcs = [
        "block_map.cc",
    ],
    hdrs = [
        "block_map.h",
    ],
    deps = [
        ":check_macros",
        ":opt_set",
        ":size_util",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "blocking_counter",
    srcs = [
        "blocking_counter.cc",
    ],
    hdrs = [
        "blocking_counter.h",
    ],
    deps = [
        ":check_macros",
        ":time",
    ],
)

cc_library(
    name = "thread_pool",
    srcs = [
        "thread_pool.cc",
    ],
    hdrs = [
        "thread_pool.h",
    ],
    deps = [
        ":blocking_counter",
        ":check_macros",
        ":time",
    ],
)

cc_library(
    name = "detect_dotprod",
    srcs = [
        "detect_dotprod.cc",
    ],
    hdrs = [
        "detect_dotprod.h",
    ],
)

cc_library(
    name = "path",
    hdrs = ["path.h"],
    deps = [":size_util"],
)

cc_library(
    name = "trace",
    srcs = [
        "trace.cc",
    ],
    hdrs = [
        "trace.h",
    ],
    deps = [
        ":block_map",
        ":check_macros",
        ":common",
        ":time",
    ],
)

cc_library(
    name = "context",
    srcs = [
        "context.cc",
    ],
    hdrs = [
        "context.h",
    ],
    deps = [
        ":allocator",
        ":check_macros",
        ":detect_dotprod",
        ":path",
        ":thread_pool",
        ":trace",
        ":tune",
    ],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    deps = [":check_macros"],
)

cc_library(
    name = "common",
    hdrs = [
        "common.h",
    ],
    deps = [
        ":check_macros",
        ":matrix",
        ":opt_set",
        ":path",
        ":size_util",
    ],
)

cc_library(
    name = "kernel",
    srcs = [
        "kernel.cc",
    ],
    hdrs = [
        "kernel.h",
    ],
    deps = [
        ":common",
        ":matrix",
        ":opt_set",
        ":path",
        ":size_util",
        ":spec",
        ":tune",
        "@gemmlowp//:fixedpoint",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "pack",
    srcs = [
        "pack.cc",
    ],
    hdrs = [
        "pack.h",
    ],
    deps = [
        ":common",
        ":matrix",
        ":opt_set",
        ":path",
        ":spec",
        ":tune",
        "@gemmlowp//:profiler",
    ],
)

load(":ruy_visibility.bzl", "ruy_visibility")

# The main library.
cc_library(
    name = "ruy",
    srcs = [
        "dispatch.h",
        "impl.h",
    ],
    hdrs = ["ruy.h"],
    visibility = ruy_visibility(),
    deps = [
        ":allocator",
        ":block_map",
        ":check_macros",
        ":common",
        ":context",
        ":kernel",
        ":matrix",
        ":opt_set",
        ":pack",
        ":path",
        ":size_util",
        ":spec",
        ":thread_pool",
        ":trace",
        ":tune",
        "@gemmlowp//:profiler",
    ],
)

# Just a usage example.
cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = [
        ":ruy",
    ],
)

# Small library to query PMU counters, for benchmark only
cc_library(
    name = "pmu",
    testonly = True,
    srcs = ["pmu.cc"],
    hdrs = ["pmu.h"],
)

load(":ruy_test_ext.bzl", "ruy_test_ext_defines", "ruy_test_ext_deps")

# Testing framework.
cc_library(
    name = "test_lib",
    testonly = True,
    hdrs = ["test.h"],
    # need defines, not copts, because it's controlling a header, test.h
    defines = ruy_test_ext_defines(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":pmu",
        ":ruy",
        ":time",
        "@com_google_googletest//:gtest",
    ] + ruy_test_ext_deps(),
)

load(":ruy_test.bzl", "ruy_benchmark", "ruy_benchmark_opt_sets", "ruy_test")

ruy_benchmark(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
    ],
)

ruy_test(
    name = "test_fast",
    srcs = ["test_fast.cc"],
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("f64", "f32", "f64", "f32"),
        ("f32", "f64", "f64", "f64"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("i8", "u8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
    ],
)

ruy_test(
    name = "test_slow",
    srcs = ["test_slow.cc"],
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
    ],
)

ruy_test(
    name = "test_special_specs",
    srcs = ["test_special_specs.cc"],
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("u8", "u8", "i32", "i16"),
    ],
)

ruy_benchmark_opt_sets(
    name = "benchmark_opt_set",
    srcs = ["benchmark.cc"],
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
    ],
    opt_sets = [
        "1",
        "3",
        "7",
        "f",
        "1f",
        "3f",
        "7f",
        "ff",
        "1ff",
        "3ff",
        "7ff",
    ],
)
