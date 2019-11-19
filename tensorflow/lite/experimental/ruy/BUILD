# Ruy is not BLAS

# TODO(b/123403203) actually make TFLite use ruy.

load(":build_defs.bzl", "ruy_copts_avx2", "ruy_copts_base", "ruy_copts_skylake", "ruy_visibility")
load(":ruy_test_ext.bzl", "ruy_test_ext_defines", "ruy_test_ext_deps")
load(":ruy_test.bzl", "ruy_benchmark", "ruy_benchmark_opt_sets", "ruy_test")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

config_setting(
    name = "optimized",
    values = {
        "compilation_mode": "opt",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
    copts = ruy_copts_base(),
)

cc_library(
    name = "check_macros",
    hdrs = ["check_macros.h"],
    copts = ruy_copts_base(),
)

cc_test(
    name = "check_macros_test",
    srcs = ["check_macros_test.cc"],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "opt_set",
    hdrs = ["opt_set.h"],
    copts = ruy_copts_base(),
)

cc_library(
    name = "time",
    hdrs = ["time.h"],
    copts = ruy_copts_base(),
)

cc_library(
    name = "wait",
    srcs = ["wait.cc"],
    hdrs = ["wait.h"],
    copts = ruy_copts_base(),
    deps = [":time"],
)

cc_test(
    name = "wait_test",
    srcs = ["wait_test.cc"],
    deps = [
        ":platform",
        ":wait",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "size_util",
    hdrs = ["size_util.h"],
    copts = ruy_copts_base(),
    deps = [":check_macros"],
)

cc_test(
    name = "size_util_test",
    srcs = ["size_util_test.cc"],
    deps = [
        ":size_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tune",
    srcs = [
        "tune.cc",
    ],
    hdrs = [
        "tune.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":opt_set",
        ":platform",
        ":time",
    ],
)

cc_library(
    name = "prepacked_cache",
    srcs = [
        "prepacked_cache.cc",
    ],
    hdrs = [
        "prepacked_cache.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":allocator",
        ":matrix",
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

cc_test(
    name = "prepacked_cache_test",
    srcs = ["prepacked_cache_test.cc"],
    deps = [
        ":prepacked_cache",
        ":ruy",
        ":time",
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
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":opt_set",
        ":path",
        ":side_pair",
        ":size_util",
        "@gemmlowp//:profiler",
    ],
)

cc_test(
    name = "block_map_test",
    srcs = ["block_map_test.cc"],
    deps = [
        ":block_map",
        ":path",
        "@com_google_googletest//:gtest",
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
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
    visibility = ruy_visibility(),
    deps = [
        ":blocking_counter",
        ":check_macros",
        ":wait",
    ],
)

cc_library(
    name = "detect_arm",
    srcs = [
        "detect_arm.cc",
    ],
    hdrs = [
        "detect_arm.h",
    ],
    copts = ruy_copts_base(),
    visibility = ruy_visibility(),
)

cc_library(
    name = "detect_x86",
    srcs = [
        "detect_x86.cc",
    ],
    hdrs = [
        "detect_x86.h",
    ],
    copts = ruy_copts_base(),
    visibility = ruy_visibility(),
    deps = [
        ":platform",
    ],
)

cc_library(
    name = "path",
    hdrs = ["path.h"],
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
    deps = [
        ":block_map",
        ":check_macros",
        ":side_pair",
        ":time",
    ],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    copts = ruy_copts_base(),
    visibility = ruy_visibility(),
    deps = [":check_macros"],
)

cc_library(
    name = "spec",
    hdrs = ["spec.h"],
    copts = ruy_copts_base(),
    visibility = ruy_visibility(),
    deps = [":matrix"],
)

cc_library(
    name = "internal_matrix",
    hdrs = ["internal_matrix.h"],
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
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
    name = "pack_common",
    hdrs = [
        "pack.h",
        "pack_arm.h",
        "pack_common.h",
        "pack_x86.h",
    ],
    copts = ruy_copts_base(),
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
    name = "kernel_arm",
    srcs = [
        "kernel_arm32.cc",
        "kernel_arm64.cc",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":common",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "pack_arm",
    srcs = [
        "pack_arm.cc",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":common",
        ":opt_set",
        ":pack_common",
        ":platform",
        "@gemmlowp//:profiler",
    ],
)

# AVX-512 compilation units.
#
# These must use the same compiler options.
RUY_COPTS_BUILT_FOR_AVX512 = ruy_copts_base() + ruy_copts_skylake()

cc_library(
    name = "kernel_avx512",
    srcs = [
        "kernel_avx512.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX512,
    deps = [
        ":check_macros",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "pack_avx512",
    srcs = [
        "pack_avx512.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX512,
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
    name = "have_built_path_for_avx512",
    srcs = [
        "have_built_path_for_avx512.cc",
    ],
    hdrs = [
        "have_built_path_for.h",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX512,
    deps = [
        ":opt_set",
        ":platform",
    ],
)
# End: AVX-512 compilation units.

# AVX2 compilation units.
#
# These must use the same compiler options.
RUY_COPTS_BUILT_FOR_AVX2 = ruy_copts_base() + ruy_copts_avx2()

cc_library(
    name = "kernel_avx2",
    srcs = [
        "kernel_avx2.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX2,
    deps = [
        ":check_macros",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "pack_avx2",
    srcs = [
        "pack_avx2.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX2,
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
    name = "have_built_path_for_avx2",
    srcs = [
        "have_built_path_for_avx2.cc",
    ],
    hdrs = [
        "have_built_path_for.h",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX2,
    deps = [
        ":opt_set",
        ":platform",
    ],
)
# End: AVX2 compilation units.

cc_library(
    name = "kernel",
    hdrs = [
        "kernel.h",
        "kernel_common.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":common",
        ":internal_matrix",
        ":kernel_arm",  # fixdeps: keep
        ":kernel_avx2",  # fixdeps: keep
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
    name = "pack",
    hdrs = [
        "pack.h",
        "pack_common.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":common",
        ":internal_matrix",
        ":matrix",
        ":opt_set",
        ":pack_arm",  # fixdeps: keep
        ":pack_avx2",  # fixdeps: keep
        ":pack_avx512",  # fixdeps: keep
        ":pack_common",
        ":path",
        ":platform",
        ":tune",
        "@gemmlowp//:profiler",
    ],
)

cc_library(
    name = "have_built_path_for",
    hdrs = [
        "have_built_path_for.h",
    ],
    deps = [
        ":have_built_path_for_avx2",
        ":have_built_path_for_avx512",
        ":platform",
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
    copts = ruy_copts_base(),
    visibility = ruy_visibility(),
    deps = [
        ":allocator",
        ":check_macros",
        ":detect_arm",
        ":detect_x86",
        ":have_built_path_for",
        ":path",
        ":platform",
        ":prepacked_cache",
        ":thread_pool",
        ":trace",
        ":tune",
    ],
)

cc_test(
    name = "context_test",
    srcs = ["context_test.cc"],
    deps = [
        ":context",
        ":path",
        ":platform",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "trmul_params",
    hdrs = ["trmul_params.h"],
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
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
        ":prepacked_cache",
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
    copts = ruy_copts_base(),
    deps = [":check_macros"],
)

# Testing framework.
cc_library(
    name = "test_lib",
    testonly = True,
    hdrs = ["test.h"],
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("f64", "f32", "f64", "f32"),
        ("f32", "f64", "f64", "f64"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("i8", "u8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
        ("i8", "i8", "i32", "i32"),
        ("i8", "u8", "i32", "i32"),
    ],
)

ruy_test(
    name = "test_slow",
    srcs = ["test_slow.cc"],
    copts = ruy_copts_base(),
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
    copts = ruy_copts_base(),
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("u8", "u8", "i32", "i16"),
    ],
)

ruy_benchmark_opt_sets(
    name = "benchmark_opt_set",
    srcs = ["benchmark.cc"],
    copts = ruy_copts_base(),
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
