load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//tensorflow:tensorflow.bzl", "if_google", "if_not_windows", "if_oss", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_cc_shared_object", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts_no_undefined", "tflite_self_contained_libs_test_suite")
load("//tensorflow/lite:special_rules.bzl", "SPECIAL_RULES_DEPS", "internal_visibility_allowlist", "tflite_internal_cc_3p_api_deps_src_all_visibility_allowlist", "tflite_portable_test_suite")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "alias_with_tflite", "cc_library_with_tflite", "cc_test_with_tflite")

package_group(
    name = "tf_lite_runtime",
    packages = ["//tensorflow/lite/..."],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "testdata/*.bin",
    "testdata/*.pb",
    "testdata/*.tflite",
    "testdata/*.csv",
    "models/testdata/*",
]) + [
    "create_op_resolver.h",
    "create_op_resolver_with_selected_ops.cc",
])

# Config to keep symbol tables even for optimized builds. Debug builds already
# preserve symbols.
config_setting(
    name = "tflite_keep_symbols",
    define_values = {"tflite_keep_symbols": "true"},
)

config_setting(
    name = "gemmlowp_profiling",
    values = {
        "copt": "-DGEMMLOWP_PROFILING",
    },
)

config_setting(
    name = "tflite_use_simple_memory_planner",
    values = {
        "copt": "-DTFLITE_USE_SIMPLE_MEMORY_PLANNER",
    },
)

config_setting(
    name = "mips",
    values = {
        "cpu": "mips",
    },
)

config_setting(
    name = "mips64",
    values = {
        "cpu": "mips64",
    },
)

# Without "cpu":"k8", when building with --copt=-DTF_LITE_STATIC_MEMORY, we get
# the following error:
# Multiple matches are not allowed unless one is unambiguously more specialized.
#
# The reason for this is that some of the tflite BUILD files (e.g.
# kernels/internal/BUILD) have config_settings based on "cpu":"k8" and the
# tf_lite_static_memory config_setting needs to be more specialized. It may be
# possible to change the existing config_settings to allow for
# tf_lite_static_memory to not require "cpu":"k8". We are not attempting that
# since we currently only using the BUILD files for x86.
config_setting(
    name = "tf_lite_static_memory",
    values = {
        "copt": "-DTF_LITE_STATIC_MEMORY",
        "cpu": "k8",
    },
)

# Flag to disable mmap: Fuchsia's minfs does not support mmap (blobfs
# does, which is why we don't always disable mmap on Fuchsia).
# TODO(b/227339138): dynamically decide depending on the filesystem.
bool_flag(
    name = "enable_fuchsia_mmap",
    build_setting_default = True,
)

config_setting(
    name = "fuchsia_mmap_disabled",
    constraint_values = if_google(
        ["//third_party/bazel_platforms/os:fuchsia"],
        [],
    ),
    flag_values = {
        ":enable_fuchsia_mmap": "False",
    },
    values = if_oss(
        # TODO(b/149248802) When we have a Fuchsia Bazel SDK update to use the values it sets.
        {"cpu": "fuchsia"},
        {},
    ),
)

config_setting(
    name = "tflite_mmap_disabled_true",
    values = {
        "copt": "-DTFLITE_MMAP_DISABLED",
    },
)

selects.config_setting_group(
    name = "tflite_mmap_disabled",
    match_any = [
        ":fuchsia_mmap_disabled",
        ":tflite_mmap_disabled_true",
        "//tensorflow:windows",
    ],
)

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src_all",
    srcs = [
        ":tflite_internal_cc_3p_api_deps_src",
        "//tensorflow/lite/core:macros.h",
        "//tensorflow/lite/core/acceleration/configuration/c:tflite_internal_cc_3p_api_deps_src",
        "//tensorflow/lite/core/api:tflite_internal_cc_3p_api_deps_src",
        "//tensorflow/lite/core/c:tflite_internal_cc_3p_api_deps_src",
        "//tensorflow/lite/kernels:tflite_internal_cc_3p_api_deps_src",
        "//tensorflow/lite/kernels/internal:tflite_internal_cc_3p_api_deps_src",
        "//tensorflow/lite/schema:tflite_internal_cc_3p_api_deps_src",
    ],
    visibility = tflite_internal_cc_3p_api_deps_src_all_visibility_allowlist(),
)

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        ":allocation.cc",
        ":allocation.h",
        ":array.cc",
        ":array.h",
        ":builtin_ops.h",
        ":logger.h",
        ":minimal_logging.cc",
        ":minimal_logging.h",
        ":minimal_logging_android.cc",
        ":mmap_allocation.cc",
        ":mutable_op_resolver.cc",
        ":mutable_op_resolver.h",
        ":op_resolver.h",
        ":portable_type_to_tflitetype.h",
        ":stderr_reporter.cc",
        ":stderr_reporter.h",
        ":tensorflow_profiler_logger.h",
        ":tensorflow_profiler_logger_shim.cc",
        ":type_to_tflitetype.h",
        ":util.cc",
        ":util.h",
    ],
    visibility = [
        "//visibility:private",
    ],
)

STABLE_FRAMEWORK_LIB_HDRS = [
    "allocation.h",
    "context_util.h",
    "//tensorflow/lite/core:macros.h",
    "error_reporter.h",
    "interpreter.h",
    "model.h",
    "model_builder.h",
    "interpreter_builder.h",
    "signature_runner.h",
    "mutable_op_resolver.h",
    "op_resolver.h",
    "stderr_reporter.h",
]

FRAMEWORK_LIB_HDRS = STABLE_FRAMEWORK_LIB_HDRS + [
    "context.h",
    "//tensorflow/lite/core:subgraph.h",
    "graph_info.h",
    "interpreter_options.h",
    "logger.h",
    "optional_debug_tools.h",
]

# Private header files that are included from the public API header files.
STABLE_FRAMEWORK_LIB_IMPL_HDRS = [
    "//tensorflow/lite/core:subgraph.h",
]

exports_files(
    [hdr for hdr in FRAMEWORK_LIB_HDRS if hdr[0] != "/"],
    visibility = ["//tensorflow/lite/core/shims:__subpackages__"],
)

cc_library(
    name = "version",
    srcs = ["//tensorflow/core/public:version.h"],
    hdrs = ["version.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
)

alias(
    name = "schema_fbs_version",
    actual = ":version",
    compatible_with = get_compatible_with_portable(),
    # avoid_dep tells build_cleaner to not use schema_fbs_version.
    tags = ["avoid_dep"],
)

cc_library(
    name = "arena_planner",
    srcs = ["arena_planner.cc"],
    hdrs = ["arena_planner.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        ":graph_info",
        ":memory_planner",
        ":simple_memory_arena",
        ":util",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "arena_planner_with_profiler",
    testonly = True,
    srcs = ["arena_planner.cc"],
    hdrs = ["arena_planner.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings() + ["-DTF_LITE_TENSORFLOW_PROFILER"],
    deps = [
        ":graph_info",
        ":memory_planner",
        ":simple_memory_arena_with_profiler",
        ":util",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "arena_planner_test",
    size = "small",
    srcs = ["arena_planner_test.cc"],
    tags = [
        "tflite_not_portable_android",
    ],
    deps = [
        ":arena_planner_with_profiler",
        ":builtin_ops",
        ":graph_info",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "arena_planner_subgraph_test",
    size = "small",
    srcs = ["arena_planner_subgraph_test.cc"],
    tags = [
        "tflite_not_portable_android",
    ],
    deps = [
        "//tensorflow/lite/kernels:subgraph_test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "simple_planner",
    srcs = ["simple_planner.cc"],
    hdrs = ["simple_planner.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        ":graph_info",
        ":memory_planner",
        ":util",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "simple_planner_test",
    size = "small",
    srcs = ["simple_planner_test.cc"],
    deps = [
        ":graph_info",
        ":simple_planner",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Main library. No ops are included here.
# TODO(aselle): Resolve problems preventing C99 usage.
cc_library(
    name = "context",
    hdrs = ["context.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "external_cpu_backend_context",
    srcs = ["external_cpu_backend_context.cc"],
    hdrs = ["external_cpu_backend_context.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "graph_info",
    srcs = ["graph_info.cc"],
    hdrs = ["graph_info.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        ":kernel_api",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "memory_planner",
    hdrs = ["memory_planner.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "simple_memory_arena",
    srcs = ["simple_memory_arena.cc"],
    hdrs = ["simple_memory_arena.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    deps = [
        ":macros",
        ":tensorflow_profiler_logger_shim",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "simple_memory_arena_with_profiler",
    testonly = True,
    srcs = ["simple_memory_arena.cc"],
    hdrs = ["simple_memory_arena.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings() + ["-DTF_LITE_TENSORFLOW_PROFILER"],
    deps = [
        ":macros",
        ":tensorflow_profiler_logger_shim",
        "//tensorflow/lite/core/c:common",
    ],
)

# A config for enabling tensorflow profiler in TFLite. Currently, it only supports dynamic
# allocation. Add '--define=tflite_tensorflow_profiler=true' in your build command line to use it.
config_setting(
    name = "tensorflow_profiler_config",
    define_values = {"tflite_tensorflow_profiler": "true"},
)

# This is intended to be used only by allocators.
cc_library(
    name = "tensorflow_profiler_logger_shim",
    srcs = ["tensorflow_profiler_logger_shim.cc"],
    hdrs = ["tensorflow_profiler_logger.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        ":macros",
    ],
    alwayslink = 1,
)

# copybara:uncomment_begin(google-only)
# # Benchmarks should link this library to see the actual result.
# cc_library(
#     name = "tensorflow_profiler_logger",
#     srcs = ["tensorflow_profiler_logger.cc"],
#     hdrs = ["tensorflow_profiler_logger.h"],
#     copts = tflite_copts_warnings(),
#     deps = [
#         ":macros",
#         "//base:addressmap",
#         "//base:examine_stack",
#         "//base:low_level_alloc",
#         "//base:malloc_hook",
#         "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
#         "//tensorflow/core/profiler/lib:traceme",
#         "//tensorflow/lite/core/c:common",
#         "//tensorflow/lite/kernels:kernel_util",
#         "@local_tsl//tsl/profiler/backends/cpu:traceme_recorder",
#         "@com_google_absl//absl/base",
#         "@com_google_absl//absl/debugging:stacktrace",
#         "@com_google_absl//absl/debugging:symbolize",
#         "@com_google_absl//absl/synchronization",
#     ],
#     alwayslink = 1,
# )
# copybara:uncomment_end

cc_library(
    name = "simple_memory_arena_debug_dump",
    srcs = ["simple_memory_arena_debug_dump.cc"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        ":simple_memory_arena",
    ],
    alwayslink = 1,
)

cc_library(
    name = "builtin_op_data",
    hdrs = ["builtin_op_data.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "kernel_api",
    hdrs = [
        "builtin_op_data.h",
        "builtin_ops.h",
        "context_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "builtin_ops",
    hdrs = ["builtin_ops.h"],
    compatible_with = get_compatible_with_portable(),
)

exports_files(["builtin_ops.h"])

cc_library(
    name = "string",
    hdrs = [
        "string_type.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
)

cc_library(
    name = "allocation",
    srcs = [
        "allocation.cc",
    ] + select({
        ":tflite_mmap_disabled": [
            "mmap_allocation_disabled.cc",
        ],
        "//conditions:default": [
            "mmap_allocation.cc",
        ],
    }),
    hdrs = [
        "allocation.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = ["//tensorflow/lite/core/api:error_reporter"],
)

cc_library(
    name = "model_builder",
    hdrs = ["model_builder.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = internal_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core:model_builder",
    ],
)

# The library that implements the full C++ API.
# See also 'framework' below, which is the corresponding public target.
# DEPRECATED: use 'framework_stable' or 'framework' instead.
alias_with_tflite(
    name = "framework_lib",
    actual = "framework",
    visibility = ["//tensorflow/lite:__subpackages__"],
)

# The public target for the full C++ API, including experimental APIs.
#
# Experimental APIs are functional, tested and usable in production; however,
# the corresponding API surface has not been finalized, and is subject to
# change.
alias_with_tflite(
    name = "framework",
    actual = "framework_experimental",
    compatible_with = get_compatible_with_portable(),
)

# The public target for the C++ API excluding experimental APIs.
# TODO(ahentz): investigate dependency on gemm_support requiring usage of tf_copts.
cc_library_with_tflite(
    name = "framework_stable",
    srcs = STABLE_FRAMEWORK_LIB_IMPL_HDRS,
    hdrs = STABLE_FRAMEWORK_LIB_HDRS,
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    tflite_deps = [
        ":cc_api_stable",
    ],
    deps = [
        ":allocation",
        ":array",
        ":external_cpu_backend_context",
        ":graph_info",
        ":kernel_api",
        ":macros",
        ":memory_planner",
        ":mutable_op_resolver",
        ":stderr_reporter",
        ":string",
        ":type_to_tflitetype",
        ":util",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core:signature_runner",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:verifier",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/profiling:root_profiler",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

# The full C++ API, including experimental APIs.
#
# Experimental APIs are functional, tested and usable in production; however,
# the corresponding API surface has not been finalized, and is subject to
# change.
#
# Note that if you have code which depends on both stable and experimental API
# features, it's fine to depend only on 'framework_experimental', since
# that includes 'framework_stable' as a subset.
cc_library_with_tflite(
    name = "framework_experimental",
    srcs = [],
    hdrs = [hdr for hdr in FRAMEWORK_LIB_HDRS if hdr != "optional_debug_tools.h"] + ["optional_debug_tools.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    tflite_deps = [
        ":cc_api_experimental",
    ],
    deps = [
        ":allocation",
        ":array",
        ":external_cpu_backend_context",
        ":graph_info",
        ":kernel_api",
        ":macros",
        ":memory_planner",
        ":mutable_op_resolver",
        ":optional_debug_tools",
        ":stderr_reporter",
        ":string",
        ":type_to_tflitetype",
        ":util",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core:framework_experimental",
        "//tensorflow/lite/core:model_builder",
        "//tensorflow/lite/core:signature_runner",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:verifier",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/profiling:root_profiler",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

cc_library(
    name = "logger",
    srcs = ["logger.cc"],
    hdrs = ["logger.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = [
        ":minimal_logging",
    ],
)

# The key parts of the C++ API, excluding experimental APIs.
# This target defines the TF Lite classes for loading models and interpreting them.
#
# This target has restricted visibility; for a public target that exposes
# these APIs, see 'framework_stable' above.
#
cc_library_with_tflite(
    name = "cc_api_stable",
    srcs = STABLE_FRAMEWORK_LIB_IMPL_HDRS,
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "interpreter_options.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    visibility = [
        "//research/drishti/benchmarking/async:__subpackages__",
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/lite/core/kernels:__subpackages__",
        "//tensorflow/lite/core/shims:__subpackages__",
        "//tensorflow/lite/delegates/flex:__subpackages__",
        "//tensorflow/lite/kernels:__subpackages__",
    ],
    deps = [
        ":allocation",
        ":array",
        ":external_cpu_backend_context",
        ":graph_info",
        ":logger",
        ":macros",
        ":memory_planner",
        ":minimal_logging",
        ":mutable_op_resolver",
        ":shared_library",
        ":simple_memory_arena",
        ":stderr_reporter",
        ":string",
        ":tensorflow_profiler_logger_shim",
        ":type_to_tflitetype",
        ":util",
        ":version",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core:model_builder",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:verifier",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates:telemetry",
        "//tensorflow/lite/delegates/xnnpack:tflite_with_xnnpack_qs8",
        "//tensorflow/lite/delegates/xnnpack:tflite_with_xnnpack_qu8",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/profiling:platform_profiler",
        "//tensorflow/lite/profiling:root_profiler",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
        "@ruy//ruy:denormal",
    ],
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

cc_library_with_tflite(
    name = "signature_runner",
    hdrs = ["signature_runner.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite/c:__subpackages__",
        "//tensorflow/lite/core:__subpackages__",
        "//tensorflow/lite/java:__subpackages__",
    ],
    deps = ["//tensorflow/lite/core:signature_runner"],
)

# The key parts of the C++ API, including experimental APIs.
#
# This target has restricted visibility; for a public target that exposes
# these APIs, see 'framework_experimental' above.
cc_library_with_tflite(
    name = "cc_api_experimental",
    srcs = [],
    hdrs = [
        "graph_info.h",
        "interpreter.h",
        "interpreter_builder.h",
        "interpreter_options.h",
        "logger.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
        "//tensorflow/lite/core:subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        ":cc_api_stable",
    ],
    visibility = [
        "//tensorflow/lite/core/shims:__subpackages__",
        "//tensorflow/lite/delegates/flex:__subpackages__",
        "//tensorflow/lite/kernels:__subpackages__",
    ],
    deps = [
        ":allocation",
        ":array",
        ":builtin_ops",
        ":external_cpu_backend_context",
        ":macros",
        ":memory_planner",
        ":minimal_logging",
        ":mutable_op_resolver",
        ":stderr_reporter",
        ":string",
        ":type_to_tflitetype",
        ":util",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core:cc_api_experimental",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:verifier",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/profiling:root_profiler",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
        "@ruy//ruy:denormal",
    ],
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

cc_library(
    name = "interpreter_options_header",
    hdrs = [
        "interpreter_options.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite/core:__subpackages__",
    ],
)

cc_library(
    name = "optional_debug_tools",
    srcs = [
        "optional_debug_tools.cc",
    ],
    hdrs = ["optional_debug_tools.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":kernel_api",
        ":macros",
        "//tensorflow/lite/core:cc_api_experimental",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "error_reporter",
    hdrs = ["error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":stderr_reporter",
        "//tensorflow/lite/core/api:error_reporter",
    ],
)

cc_library(
    name = "stderr_reporter",
    srcs = ["stderr_reporter.cc"],
    hdrs = ["stderr_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":minimal_logging",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "op_resolver",
    hdrs = ["op_resolver.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":mutable_op_resolver",
        "//tensorflow/lite/core/api:op_resolver",
    ],
)

cc_library(
    name = "mutable_op_resolver",
    srcs = ["mutable_op_resolver.cc"],
    hdrs = ["mutable_op_resolver.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//tensorflow/lite/core/api:op_resolver",
        "//tensorflow/lite/core/api:op_resolver_internal",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library_with_tflite(
    name = "mutable_op_resolver_utils",
    srcs = ["mutable_op_resolver_utils.cc"],
    hdrs = ["mutable_op_resolver_utils.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    tflite_deps = [
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":builtin_ops",
        ":mutable_op_resolver",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "string_util",
    srcs = ["string_util.cc"],
    hdrs = ["string_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        ":string",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ],
)

# Link this library to inject XNNPACK delegate to TFLite runtime automatically
# by utilizing the weak symbols if they're supported by the platform.
cc_library(
    name = "tflite_with_xnnpack",
    srcs = ["tflite_with_xnnpack.cc"],
    copts = tflite_copts() + tflite_copts_warnings(),
    linkstatic = True,
    deps = [
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
    ],
    alwayslink = 1,
)

# Enables calling xnnpack impl directly from TFLite kernels.
config_setting(
    name = "tflite_kernel_use_xnnpack_false",
    define_values = {"tflite_kernel_use_xnnpack": "false"},
)

cc_library(
    name = "tflite_kernel_use_xnnpack_optional",
    compatible_with = get_compatible_with_portable(),
    defines = select({
        ":tflite_kernel_use_xnnpack_false": [],
        ":tflite_with_xnnpack_explicit_false": [],
        "//conditions:default": [
            "TFLITE_KERNEL_USE_XNNPACK",
        ],
    }),
)

# Enables applying XNNPACK delegate for float models in TFLite runtime.
# WARNING: This build flag is experimental and subject to change.
config_setting(
    name = "tflite_with_xnnpack_explicit_true",
    define_values = {"tflite_with_xnnpack": "true"},
)

config_setting(
    name = "tflite_with_xnnpack_explicit_false",
    define_values = {"tflite_with_xnnpack": "false"},
)

cc_library(
    name = "tflite_with_xnnpack_enabled",
    compatible_with = get_compatible_with_portable(),
    defines = ["TFLITE_BUILD_WITH_XNNPACK_DELEGATE"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
    ],
)

cc_library(
    name = "tflite_with_xnnpack_default",
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    # Note: adding ":tflite_with_xnnpack_enabled" to the values of following
    # configuration conditions will make TFLite interpreter to apply XNNPACK
    # delegate by default.
    deps = [
        ":tflite_with_xnnpack_enabled",
    ],
)

cc_library(
    name = "tflite_with_xnnpack_optional",
    srcs = ["tflite_with_xnnpack_optional.cc"],
    hdrs = [
        "tflite_with_xnnpack_optional.h",
        "//tensorflow/lite/core:macros.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ] + select({
        ":tflite_with_xnnpack_explicit_true": [
            ":tflite_with_xnnpack_enabled",
            "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        ],
        ":tflite_with_xnnpack_explicit_false": [],
        "//conditions:default": [
            ":tflite_with_xnnpack_default",
            "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        ],
    }),
)

cc_test(
    name = "string_util_test",
    size = "small",
    srcs = ["string_util_test.cc"],
    deps = [
        ":framework",
        ":string",
        ":string_util",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "interpreter_test_util",
    testonly = True,
    hdrs = ["interpreter_test_util.h"],
    deps = [
        ":builtin_op_data",
        ":external_cpu_backend_context",
        ":framework",
        ":string_util",
        ":version",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
    ],
)

# Test main interpreter
cc_test(
    name = "interpreter_test",
    size = "small",
    srcs = [
        "interpreter_test.cc",
    ],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    tags = [
        "tflite_smoke_test",
    ],
    deps = [
        ":external_cpu_backend_context",
        ":framework",
        ":interpreter_test_util",
        ":string",
        ":string_util",
        ":util",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

# Test graph utils
cc_test(
    name = "graph_info_test",
    size = "small",
    srcs = ["graph_info_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Test arena allocator
cc_test(
    name = "simple_memory_arena_test",
    size = "small",
    srcs = ["simple_memory_arena_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":simple_memory_arena",
        "//tensorflow/lite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

# Test model framework with the flex library linked into the target.
tf_cc_test(
    name = "model_flex_test",
    size = "small",
    srcs = ["model_flex_test.cc"],
    data = [
        "testdata/multi_add_flex.bin",
    ],
    tags = [
        "no_gpu",  # GPU + flex is not officially supported.
        "no_windows",  # No weak symbols with MSVC.
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Test model framework with the XNNPACK delegate.
cc_test(
    name = "model_xnnpack_test",
    size = "small",
    srcs = [
        "model_xnnpack_test.cc",
    ],
    data = [
        "testdata/multi_add.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
        "tflite_smoke_test",
    ],
    deps = [
        ":framework",
        ":string",
        ":tflite_with_xnnpack",
        ":util",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "allocation_test",
    size = "small",
    srcs = ["allocation_test.cc"],
    data = [
        "testdata/empty_model.bin",
    ],
    tags = [
        "tflite_smoke_test",
    ],
    deps = [
        ":allocation",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Test OpResolver.
cc_test(
    name = "mutable_op_resolver_test",
    size = "small",
    srcs = ["mutable_op_resolver_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_tflite(
    name = "mutable_op_resolver_utils_test",
    srcs = ["mutable_op_resolver_utils_test.cc"],
    tags = ["tflite_disable_mobile_test"],  # Due to b/144306101 and b/329899620.
    tflite_deps = [
        ":mutable_op_resolver_utils",
        ":test_util",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
    ],
    deps = [
        ":builtin_ops",
        ":mutable_op_resolver",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "stderr_reporter_test",
    srcs = ["stderr_reporter_test.cc"],
    deps = [
        ":stderr_reporter",
        "//tensorflow/lite/core/api:error_reporter",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "optional_debug_tools_test",
    size = "small",
    srcs = ["optional_debug_tools_test.cc"],
    data = ["testdata/add.bin"],
    deps = [
        ":framework",
        ":optional_debug_tools",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings() + tflite_copts(),
    deps = [
        ":array",
        ":kernel_api",
        ":macros",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings() + tflite_copts(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "array_test",
    srcs = ["array_test.cc"],
    deps = [
        ":array",
        "//tensorflow/lite/core/c:common",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "test_util_test",
    srcs = ["test_util_test.cc"],
    deps = [
        ":array",
        ":test_util",
        "//tensorflow/lite/core/c:common",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "create_op_resolver_header",
    hdrs = ["create_op_resolver.h"],
    deps = ["//tensorflow/lite/core:create_op_resolver_header"],
)

# Defines CreateOpResolver with all builtin ops.
cc_library_with_tflite(
    name = "create_op_resolver_with_builtin_ops",
    hdrs = ["create_op_resolver.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = ["//tensorflow/lite/core:create_op_resolver_with_builtin_ops"],
)

# Defines CreateOpResolver with a selected subset of ops.
cc_library_with_tflite(
    name = "create_op_resolver_with_selected_ops",
    srcs = ["create_op_resolver_with_selected_ops.cc"],
    hdrs = ["create_op_resolver.h"],
    copts = tflite_copts(),
    # This library is not self-contained; it needs to be linked
    # together with an implementation of RegisterSelectedOps
    # (normally one automatically generated by gen_selected_ops).
    tags = ["allow_undefined_symbols"],
    deps = [
        ":mutable_op_resolver",
        ":op_resolver",
        "//tensorflow/lite/core:create_op_resolver_header",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
    # Some targets only have an implicit dependency on CreateOpResolver.
    # This avoids warnings about backwards references when linking.
    alwayslink = True,
)

cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "minimal_logging",
    srcs = [
        "minimal_logging.cc",
    ] + select({
        "//tensorflow:android": [
            "minimal_logging_android.cc",
        ],
        "//tensorflow:ios": [
            "minimal_logging_ios.cc",
        ],
        "//tensorflow:macos": [
            "minimal_logging_default.cc",
        ],
        "//conditions:default": [
            "minimal_logging_default.cc",
        ],
    }),
    hdrs = [
        "logger.h",
        "minimal_logging.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings() + tflite_copts(),
    linkopts = select({
        "//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    visibility = internal_visibility_allowlist(),
)

cc_library(
    name = "type_to_tflitetype",
    hdrs = [
        "portable_type_to_tflitetype.h",
    ] + select({
        ":tf_lite_static_memory": [],
        "//conditions:default": [
            "type_to_tflitetype.h",
        ],
    }),
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ],
)

cc_test(
    name = "type_to_tflitetype_test",
    size = "small",
    srcs = ["type_to_tflitetype_test.cc"],
    deps = [
        ":type_to_tflitetype",
        "//tensorflow/lite/core/c:c_api_types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "minimal_logging_test",
    size = "small",
    srcs = ["minimal_logging_test.cc"],
    deps = [
        ":minimal_logging",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "shared_library",
    hdrs = ["shared_library.h"],
    compatible_with = get_compatible_with_portable(),
    linkopts = if_not_windows(["-ldl"]),
)

cc_library(
    name = "macros",
    hdrs = ["//tensorflow/lite/core:macros.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "stateful_error_reporter",
    hdrs = ["stateful_error_reporter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["//tensorflow/lite/core/api:error_reporter"],
)

# Shared lib target for convenience, pulls in the core runtime and builtin ops.
# Note: This target is not yet finalized, and the exact set of exported (C/C++)
# APIs is subject to change. The output library name is platform dependent:
#   - Linux/Android: `libtensorflowlite.so`
#   - Mac: `libtensorflowlite.dylib`
#   - Windows: `tensorflowlite.dll`
tflite_cc_shared_object(
    name = "tensorflowlite",
    # Until we have more granular symbol export for the C++ API on Windows,
    # export all symbols.
    features = ["windows_export_all_symbols"],
    linkopts = tflite_linkopts_no_undefined() + select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
        ],
        "//tensorflow:ios": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,--version-script,$(location //tensorflow/lite:tflite_version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":framework",
        ":tflite_exported_symbols.lds",
        ":tflite_version_script.lds",
        "//tensorflow/lite/kernels:builtin_ops_all_linked",
    ],
)

cc_library_with_tflite(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    tflite_deps = [
        "//tensorflow/lite/c:test_util",
    ],
    deps = [
        ":array",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

bzl_library(
    name = "build_def_bzl",
    srcs = ["build_def.bzl"],
    visibility = [
        "//learning/brain/models/app_benchmarks:__pkg__",
        "//tensorflow/lite/core/shims:__subpackages__",
    ],
    deps = [
        ":special_rules_bzl",
        "//tensorflow:tensorflow_bzl",
        "//tensorflow/lite/java:aar_with_jni_bzl",
        "@bazel_skylib//rules:build_test",
    ],
)

# This target defines the TFLITE_CONDITIONAL_NAMESPACE macro.
cc_library_with_tflite(
    name = "namespace",
    hdrs = ["namespace.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//visibility:public",
    ],
)

bzl_library(
    name = "special_rules_bzl",
    srcs = ["special_rules.bzl"],
    visibility = ["//tensorflow/lite/core/shims:__subpackages__"],
    deps = SPECIAL_RULES_DEPS,
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")

tflite_portable_test_suite()
