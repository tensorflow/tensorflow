load("//tensorflow:tensorflow.bzl", "if_not_windows", "tf_cc_test")
load("//tensorflow/lite:build_def.bzl", "if_tflite_experimental_runtime", "tflite_cc_shared_object", "tflite_copts", "tflite_experimental_runtime_linkopts")
load("//tensorflow/lite/micro:build_def.bzl", "cc_library")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(glob([
    "testdata/*.bin",
    "testdata/*.pb",
    "testdata/*.tflite",
    "testdata/*.csv",
    "models/testdata/*",
]))

config_setting(
    name = "enable_default_profiler",
    values = {
        "copt": "-DTFLITE_ENABLE_DEFAULT_PROFILER",
    },
)

config_setting(
    name = "gemmlowp_profiling",
    values = {
        "copt": "-DGEMMLOWP_PROFILING",
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

config_setting(
    name = "tflite_experimental_runtime_eager",
    values = {"define": "tflite_experimental_runtime=eager"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "tflite_experimental_runtime_non_eager",
    values = {"define": "tflite_experimental_runtime=non-eager"},
    visibility = ["//visibility:public"],
)

TFLITE_DEFAULT_COPTS = if_not_windows([
    "-Wall",
    "-Wno-comment",
    "-Wno-extern-c-compat",
])

FRAMEWORK_LIB_HDRS = [
    "allocation.h",
    "context.h",
    "context_util.h",
    "core/macros.h",
    "core/subgraph.h",
    "error_reporter.h",
    "graph_info.h",
    "interpreter.h",
    "model.h",
    "model_builder.h",
    "interpreter_builder.h",
    "mutable_op_resolver.h",
    "op_resolver.h",
    "optional_debug_tools.h",
    "stderr_reporter.h",
]

cc_library(
    name = "version",
    hdrs = ["version.h"],
    build_for_embedded = True,
    copts = TFLITE_DEFAULT_COPTS,
    # Note that we only use the header defines from :version_lib.
    deps = ["//tensorflow/core:version_lib"],
)

# TODO(b/128420794): Migrate clients to use :version directly.
alias(
    name = "schema_fbs_version",
    actual = ":version",
)

cc_library(
    name = "arena_planner",
    srcs = ["arena_planner.cc"],
    hdrs = ["arena_planner.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = [
        ":graph_info",
        ":memory_planner",
        ":simple_memory_arena",
        "//tensorflow/lite/c:common",
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
        ":arena_planner",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Main library. No ops are included here.
# TODO(aselle): Resolve problems preventing C99 usage.
cc_library(
    name = "context",
    hdrs = ["context.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "external_cpu_backend_context",
    srcs = ["external_cpu_backend_context.cc"],
    hdrs = ["external_cpu_backend_context.h"],
    build_for_embedded = True,
    copts = TFLITE_DEFAULT_COPTS,
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "graph_info",
    hdrs = ["graph_info.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "memory_planner",
    hdrs = ["memory_planner.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "simple_memory_arena",
    srcs = ["simple_memory_arena.cc"],
    hdrs = ["simple_memory_arena.h"],
    copts = TFLITE_DEFAULT_COPTS,
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "builtin_op_data",
    hdrs = [
        "builtin_op_data.h",
    ],
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "kernel_api",
    hdrs = [
        "builtin_op_data.h",
        "builtin_ops.h",
        "context_util.h",
    ],
    deps = ["//tensorflow/lite/c:common"],
)

exports_files(["builtin_ops.h"])

cc_library(
    name = "string",
    hdrs = [
        "string_type.h",
    ],
    build_for_embedded = True,
    copts = TFLITE_DEFAULT_COPTS,
)

cc_library(
    name = "allocation",
    srcs = [
        "allocation.cc",
    ] + select({
        "//tensorflow:android": [
            "mmap_allocation.cc",
        ],
        "//tensorflow:windows": [
            "mmap_allocation_disabled.cc",
        ],
        "//conditions:default": [
            "mmap_allocation.cc",
        ],
    }),
    hdrs = [
        "allocation.h",
    ],
    copts = TFLITE_DEFAULT_COPTS,
    deps = [
        ":string",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
    ],
)

cc_library(
    name = "framework_lib",
    srcs = [
        "core/subgraph.cc",
        "graph_info.cc",
        "interpreter.cc",
        "interpreter_builder.cc",
        "model_builder.cc",
        "mutable_op_resolver.cc",
        "optional_debug_tools.cc",
        "stderr_reporter.cc",
    ],
    hdrs = FRAMEWORK_LIB_HDRS,
    copts = tflite_copts() + TFLITE_DEFAULT_COPTS,
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":allocation",
        ":arena_planner",
        ":external_cpu_backend_context",
        ":graph_info",
        ":memory_planner",
        ":minimal_logging",
        ":simple_memory_arena",
        ":string",
        ":tflite_with_xnnpack_optional",
        ":type_to_tflitetype",
        ":util",
        ":version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/schema:schema_fbs",
    ] + select({
        ":enable_default_profiler": [
            "//tensorflow/lite/profiling:platform_profiler",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

# TODO(ahentz): investigate dependency on gemm_support requiring usage of tf_copts.
cc_library(
    name = "framework",
    srcs = [
    ],
    hdrs = FRAMEWORK_LIB_HDRS,
    copts = tflite_copts() + TFLITE_DEFAULT_COPTS,
    defines = if_tflite_experimental_runtime(
        if_eager = ["TFLITE_EXPERIMENTAL_RUNTIME_EAGER"],
        if_non_eager = ["TFLITE_EXPERIMENTAL_RUNTIME_NON_EAGER"],
        if_none = [],
    ),
    deps = [
        ":framework_lib",
        ":allocation",
        ":arena_planner",
        ":external_cpu_backend_context",
        ":graph_info",
        ":memory_planner",
        ":minimal_logging",
        ":simple_memory_arena",
        ":string",
        ":type_to_tflitetype",
        ":util",
        ":version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/schema:schema_fbs",
    ] + tflite_experimental_runtime_linkopts(),
)

cc_library(
    name = "string_util",
    srcs = ["string_util.cc"],
    hdrs = ["string_util.h"],
    build_for_embedded = True,
    copts = TFLITE_DEFAULT_COPTS,
    deps = [
        ":string",
        "//tensorflow/lite/c:common",
    ],
)

# Link this library to inject XNNPACK delegate to TFLite runtime automatically
# by utilizing the weak symbols if they're supported by the platform.
cc_library(
    name = "tflite_with_xnnpack",
    srcs = ["tflite_with_xnnpack.cc"],
    copts = tflite_copts() + TFLITE_DEFAULT_COPTS,
    linkstatic = True,
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
    ],
    alwayslink = 1,
)

# Enables applying XNNPACK delegate for float models in TFLite runtime.
# WARNING: This build flag is experimental and subject to change.
config_setting(
    name = "tflite_with_xnnpack_enabled",
    values = {"define": "tflite_with_xnnpack=true"},
)

cc_library(
    name = "tflite_with_xnnpack_optional",
    srcs = ["tflite_with_xnnpack_optional.cc"],
    hdrs = [
        "core/macros.h",
        "tflite_with_xnnpack_optional.h",
    ],
    copts = tflite_copts() + TFLITE_DEFAULT_COPTS,
    defines = select({
        ":tflite_with_xnnpack_enabled": ["TFLITE_BUILD_WITH_XNNPACK_DELEGATE"],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite/c:common",
    ] + select({
        ":tflite_with_xnnpack_enabled": [
            "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "string_util_test",
    size = "small",
    srcs = ["string_util_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":framework",
        ":string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test main interpreter
cc_test(
    name = "interpreter_test",
    size = "small",
    srcs = ["interpreter_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":external_cpu_backend_context",
        ":framework",
        ":string_util",
        ":version",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "//third_party/eigen3",
        "@com_google_googletest//:gtest",
    ],
)

# Test graph utils
cc_test(
    name = "graph_info_test",
    size = "small",
    srcs = ["graph_info_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":framework",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test arena allocator
cc_test(
    name = "simple_memory_arena_test",
    size = "small",
    srcs = ["simple_memory_arena_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":simple_memory_arena",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test model framework.
cc_test(
    name = "model_test",
    size = "small",
    srcs = ["model_test.cc"],
    data = [
        "testdata/0_subgraphs.bin",
        "testdata/2_subgraphs.bin",
        "testdata/empty_model.bin",
        "testdata/multi_add_flex.bin",
        "testdata/sparse_tensor.bin",
        "testdata/test_min_runtime.bin",
        "testdata/test_model.bin",
        "testdata/test_model_broken.bin",
    ],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
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
        "no_windows",  # TODO(b/116667551): No weak symbols with MSVC.
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
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
        "no_windows",  # No weak symbols with MSVC.
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":framework",
        ":tflite_with_xnnpack",
        ":util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Test OpResolver.
cc_test(
    name = "mutable_op_resolver_test",
    size = "small",
    srcs = ["mutable_op_resolver_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":framework",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    copts = TFLITE_DEFAULT_COPTS + tflite_copts(),
    deps = [
        ":kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
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
        "//conditions:default": [
            "minimal_logging_default.cc",
        ],
    }),
    hdrs = ["minimal_logging.h"],
    copts = TFLITE_DEFAULT_COPTS + tflite_copts(),
    linkopts = select({
        "//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
)

cc_library(
    name = "type_to_tflitetype",
    hdrs = ["type_to_tflitetype.h"],
    build_for_embedded = True,
    deps = ["//tensorflow/lite/c:common"],
)

cc_test(
    name = "minimal_logging_test",
    size = "small",
    srcs = ["minimal_logging_test.cc"],
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":minimal_logging",
        "@com_google_googletest//:gtest",
    ],
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
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,-z,defs",
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

tflite_portable_test_suite()
