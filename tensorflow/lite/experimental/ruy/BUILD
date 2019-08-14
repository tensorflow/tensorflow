# Ruy is not BLAS

# TODO(b/123403203) actually make TFLite use ruy.

load(":ruy_visibility.bzl", "ruy_visibility")
load(":ruy_test_ext.bzl", "ruy_test_ext_defines", "ruy_test_ext_deps")
load(":ruy_test.bzl", "ruy_benchmark", "ruy_benchmark_opt_sets", "ruy_test")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

# 1. Enable -mfpu=neon unconditionally on ARM32. If it turns out that we need to support
#    ARM32 without NEON then we'll implement runtime detection and dispatch at that point.
# 2. Explicitly pass -O3 on mobile configs where just "-c opt" means "optimize for code size".
#    We would want to only do that when compilation_mode is "opt", but limitations of
#    the "select" keyword (no nested selects, no AND boolean) seem to make that difficult
#    at the moment. For debugging purposes, this can be overridded on the command line, e.g.
#      bazel build -c dbg --copt=-O0 ...
RUY_COPTS = select({
    "//tensorflow:android_arm64": [
        "-O3",
    ],
    "//tensorflow:android_arm": [
        "-O3",
        "-mfpu=neon",
    ],
    "//conditions:default": [
    ],
})

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
    copts = RUY_COPTS,
)

cc_library(
    name = "check_macros",
    hdrs = ["check_macros.h"],
    copts = RUY_COPTS,
    deps = ["//tensorflow/lite/kernels/internal:compatibility"],
)

cc_library(
    name = "opt_set",
    hdrs = ["opt_set.h"],
    copts = RUY_COPTS,
)

cc_library(
    name = "time",
    hdrs = ["time.h"],
    copts = RUY_COPTS,
)

cc_library(
    name = "wait",
    srcs = ["wait.cc"],
    hdrs = ["wait.h"],
    copts = RUY_COPTS,
    deps = [":time"],
)

cc_test(
    name = "wait_test",
    srcs = ["wait_test.cc"],
    deps = [
        ":wait",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "size_util",
    hdrs = ["size_util.h"],
    copts = RUY_COPTS,
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
    copts = RUY_COPTS,
    deps = [
        ":opt_set",
        ":platform",
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
    copts = RUY_COPTS,
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
    name = "side_pair",
    hdrs = ["side_pair.h"],
    copts = RUY_COPTS,
    deps = [":check_macros"],
)

cc_library(
    name = "block_map",
    srcs = [
        "block_map.cc",
    ],
    hdrs = [
        "block_map.h",
    ],
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":opt_set",
        ":side_pair",
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
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":wait",
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
    copts = RUY_COPTS,
    visibility = ruy_visibility(),
    deps = [
        ":blocking_counter",
        ":check_macros",
        ":wait",
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
    copts = RUY_COPTS,
    visibility = ruy_visibility(),
)

cc_library(
    name = "path",
    hdrs = ["path.h"],
    copts = RUY_COPTS,
    visibility = ruy_visibility(),
    deps = [
        ":platform",
        ":size_util",
    ],
)

cc_library(
    name = "trace",
    srcs = [
        "trace.cc",
    ],
    hdrs = [
        "trace.h",
    ],
    copts = RUY_COPTS,
    deps = [
        ":block_map",
        ":check_macros",
        ":side_pair",
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
    copts = RUY_COPTS,
    visibility = ruy_visibility(),
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
    copts = RUY_COPTS,
    visibility = ruy_visibility(),
    deps = [":check_macros"],
)

cc_library(
    name = "spec",
    hdrs = ["spec.h"],
    copts = RUY_COPTS,
    visibility = ruy_visibility(),
    deps = [":matrix"],
)

cc_library(
    name = "internal_matrix",
    hdrs = ["internal_matrix.h"],
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":common",
        ":matrix",
        ":size_util",
    ],
)

cc_library(
    name = "common",
    hdrs = [
        "common.h",
    ],
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":matrix",
        ":opt_set",
        ":path",
        ":platform",
    ],
)

cc_library(
    name = "kernel_common",
    hdrs = [
        "kernel.h",
        "kernel_arm.h",
        "kernel_common.h",
        "kernel_x86.h",
    ],
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":common",
        ":internal_matrix",
        ":matrix",
        ":opt_set",
        ":path",
        ":platform",
        ":side_pair",
        ":size_util",
        ":spec",
        ":tune",
        "@gemmlowp//:fixedpoint",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "kernel_arm",
    srcs = [
        "kernel_arm32.cc",
        "kernel_arm64.cc",
    ],
    copts = RUY_COPTS,
    deps = [
        ":common",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "kernel_avx512",
    srcs = [
        "kernel_avx512.cc",
    ],
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "kernel",
    hdrs = [
        "kernel.h",
        "kernel_common.h",
    ],
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":common",
        ":internal_matrix",
        ":kernel_arm",  # fixdeps: keep
        ":kernel_avx512",  # fixdeps: keep
        ":kernel_common",
        ":matrix",
        ":opt_set",
        ":path",
        ":platform",
        ":side_pair",
        ":size_util",
        ":spec",
        ":tune",
        "@gemmlowp//:fixedpoint",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "pack_common",
    hdrs = [
        "pack.h",
        "pack_arm.h",
        "pack_common.h",
        "pack_x86.h",
    ],
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":common",
        ":internal_matrix",
        ":matrix",
        ":opt_set",
        ":path",
        ":platform",
        ":tune",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "pack_arm",
    srcs = [
        "pack_arm.cc",
    ],
    copts = RUY_COPTS,
    deps = [
        ":common",
        ":opt_set",
        ":pack_common",
        ":platform",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "pack_avx512",
    srcs = [
        "pack_avx512.cc",
    ],
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":matrix",
        ":opt_set",
        ":pack_common",
        ":path",
        ":platform",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "pack",
    hdrs = [
        "pack.h",
        "pack_common.h",
    ],
    copts = RUY_COPTS,
    deps = [
        ":check_macros",
        ":common",
        ":internal_matrix",
        ":matrix",
        ":opt_set",
        ":pack_arm",  # fixdeps: keep
        ":pack_avx512",  # fixdeps: keep
        ":pack_common",
        ":path",
        ":platform",
        ":tune",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "trmul_params",
    hdrs = ["trmul_params.h"],
    copts = RUY_COPTS,
    deps = [
        ":internal_matrix",
        ":side_pair",
        ":tune",
    ],
)

cc_library(
    name = "trmul",
    srcs = ["trmul.cc"],
    hdrs = ["trmul.h"],
    copts = RUY_COPTS,
    deps = [
        ":allocator",
        ":block_map",
        ":check_macros",
        ":common",
        ":context",
        ":internal_matrix",
        ":matrix",
        ":opt_set",
        ":side_pair",
        ":size_util",
        ":spec",
        ":thread_pool",
        ":trace",
        ":trmul_params",
        ":tune",
        "@gemmlowp//:profiler",
    ],
)

# The main library.
cc_library(
    name = "ruy",
    srcs = [
        "dispatch.h",
        "prepack.h",
    ],
    hdrs = [
        "ruy.h",
        "ruy_advanced.h",
    ],
    copts = RUY_COPTS,
    visibility = ruy_visibility(),
    deps = [
        ":check_macros",
        ":common",
        ":context",
        ":internal_matrix",
        ":kernel",
        ":matrix",
        ":opt_set",
        ":pack",
        ":path",
        ":side_pair",
        ":size_util",
        ":spec",
        ":trmul",
        ":trmul_params",
        ":tune",
        "@gemmlowp//:profiler",
    ],
)

# Usage examples.
cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = [":ruy"],
)

# Usage examples of the advanced API.
cc_binary(
    name = "example_advanced",
    srcs = ["example_advanced.cc"],
    deps = [":ruy"],
)

# Small library to query PMU counters, for benchmark only
cc_library(
    name = "pmu",
    testonly = True,
    srcs = ["pmu.cc"],
    hdrs = ["pmu.h"],
    copts = RUY_COPTS,
    deps = [":check_macros"],
)

# Testing framework.
cc_library(
    name = "test_lib",
    testonly = True,
    hdrs = ["test.h"],
    copts = RUY_COPTS,
    # need defines, not copts, because it's controlling a header, test.h
    defines = ruy_test_ext_defines(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":matrix",
        ":pmu",
        ":ruy",
        ":spec",
        ":time",
        "@com_google_googletest//:gtest",
        ":platform",
    ] + ruy_test_ext_deps(),
)

ruy_benchmark(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    copts = RUY_COPTS,
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
        ("i8", "i8", "i32", "i32"),
    ],
)

ruy_test(
    name = "test_fast",
    srcs = ["test_fast.cc"],
    copts = RUY_COPTS,
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("f64", "f32", "f64", "f32"),
        ("f32", "f64", "f64", "f64"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("i8", "u8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
        ("i8", "i8", "i32", "i32"),
    ],
)

ruy_test(
    name = "test_slow",
    srcs = ["test_slow.cc"],
    copts = RUY_COPTS,
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
        ("i8", "i8", "i32", "i32"),
    ],
    tags = ["slow"],
)

ruy_test(
    name = "test_special_specs",
    srcs = ["test_special_specs.cc"],
    copts = RUY_COPTS,
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("u8", "u8", "i32", "i16"),
    ],
)

ruy_benchmark_opt_sets(
    name = "benchmark_opt_set",
    srcs = ["benchmark.cc"],
    copts = RUY_COPTS,
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

tflite_portable_test_suite()
