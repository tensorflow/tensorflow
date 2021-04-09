# Description: this package contains shim library targets that forward
# to the TF Lite C and C++ API targets.  See README.md.

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load(":build_defs.bzl", "build_test")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

#------------------------------------------------------------------------------
# C++ API

FRAMEWORK_LIB_HDRS = [
    "//tensorflow/lite:allocation.h",
    "//tensorflow/lite:context.h",
    "//tensorflow/lite:context_util.h",
    "//tensorflow/lite:core/macros.h",
    "//tensorflow/lite:core/subgraph.h",
    "//tensorflow/lite:error_reporter.h",
    "//tensorflow/lite:graph_info.h",
    "//tensorflow/lite:mutable_op_resolver.h",
    "//tensorflow/lite:op_resolver.h",
    "//tensorflow/lite:optional_debug_tools.h",
    "//tensorflow/lite:stderr_reporter.h",
]

CC_API_HDRS = [
    "cc/interpreter.h",
    "cc/interpreter_builder.h",
    "cc/model.h",
    "cc/model_builder.h",
]

cc_library(
    name = "framework",
    srcs = [],
    hdrs = FRAMEWORK_LIB_HDRS + CC_API_HDRS,
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:arena_planner",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:framework_lib",
        "//tensorflow/lite:graph_info",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:memory_planner",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:simple_memory_arena",
        "//tensorflow/lite:string",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite:util",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:verifier",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "cc_api",
    srcs = [],
    hdrs = CC_API_HDRS,
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//tensorflow/lite:__pkg__"],
    deps = [
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:arena_planner",
        "//tensorflow/lite:cc_api",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:graph_info",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:memory_planner",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:simple_memory_arena",
        "//tensorflow/lite:string",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite:util",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:verifier",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "builtin_ops",
    hdrs = [
        "cc/kernels/register.h",
        "//tensorflow/lite/kernels:builtin_op_kernels.h",
        "//tensorflow/lite/kernels:fully_connected.h",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:cc_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)

build_test(
    name = "cc_api_build_test",
    targets = [
        ":builtin_ops",
        ":cc_api",
        ":framework",
    ],
)

cc_library(
    name = "delegate_registry",
    hdrs = ["cc/experimental/acceleration/configuration/delegate_registry.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/experimental/acceleration/configuration:delegate_registry",
    ],
)

build_test(
    name = "delegate_registry_build_test",
    targets = [
        ":delegate_registry",
    ],
)

cc_library(
    name = "verifier",
    hdrs = ["cc/tools/verifier.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/tools:verifier",
    ],
)

build_test(
    name = "verifier_build_test",
    targets = [
        ":verifier",
    ],
)

#------------------------------------------------------------------------------
# C API

cc_library(
    name = "c_api",
    hdrs = ["c/c_api.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = ["//tensorflow/lite/c:c_api"],
)

cc_library(
    name = "c_api_experimental",
    hdrs = ["c/c_api_experimental.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = ["//tensorflow/lite/c:c_api_experimental"],
)

cc_library(
    name = "common",
    hdrs = ["c/common.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "builtin_op_data",
    hdrs = ["c/builtin_op_data.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = ["//tensorflow/lite/c:common"],
)

build_test(
    name = "c_api_build_test",
    targets = [
        ":builtin_op_data",
        ":c_api",
        ":c_api_experimental",
        ":common",
    ],
)

#------------------------------------------------------------------------------
# Testing infrastructure

cc_library(
    name = "c_shims_test_util",
    testonly = True,
    srcs = ["c/shims_test_util.cc"],
    hdrs = ["c/shims_test_util.h"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
)

cc_library(
    name = "cc_shims_test_util",
    testonly = True,
    hdrs = ["cc/shims_test_util.h"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        ":c_shims_test_util",
        "@com_google_googletest//:gtest",
    ],
)

build_test(
    name = "shims_test_util_build_test",
    targets = [
        ":c_shims_test_util",
        ":cc_shims_test_util",
    ],
)

#------------------------------------------------------------------------------
# JNI bindings (Java API and Java Tasks library)

# Contains code to initialize TFLite through JNI in the internal version.
cc_library(
    name = "jni_initialization",
    srcs = [],
    # Prevent automated tools from removing this target as a dependency due to
    # it being empty.
    tags = ["keep_dep"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
)

#------------------------------------------------------------------------------

tflite_portable_test_suite()
