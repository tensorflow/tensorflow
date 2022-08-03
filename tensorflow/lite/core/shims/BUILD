# Description: this package contains shim library targets that forward
# to the TF Lite C and C++ API targets.  See README.md.

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(":build_defs.bzl", "build_test")
load(":cc_library_with_tflite.bzl", "custom_c_library_with_tflite")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(
    srcs = ["c/c_api.h"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
)

#------------------------------------------------------------------------------
# C++ API

FRAMEWORK_STABLE_LIB_HDRS = [
    "//tensorflow/lite:allocation.h",
    "//tensorflow/lite:context_util.h",
    "//tensorflow/lite:core/macros.h",
    "//tensorflow/lite:error_reporter.h",
    "//tensorflow/lite:mutable_op_resolver.h",
    "//tensorflow/lite:op_resolver.h",
    "//tensorflow/lite:stderr_reporter.h",
]

FRAMEWORK_LIB_HDRS = FRAMEWORK_STABLE_LIB_HDRS + [
    "//tensorflow/lite:context.h",
    "//tensorflow/lite:core/subgraph.h",
    "//tensorflow/lite:graph_info.h",
    "//tensorflow/lite:optional_debug_tools.h",
]

CC_API_HDRS = [
    "cc/interpreter.h",
    "cc/interpreter_builder.h",
    "cc/model.h",
    "cc/model_builder.h",
]

# The public target for the full C++ API.
#
# Experimental APIs are functional, tested and usable in production; however,
# the corresponding API surface has not been finalized, and is subject to
# change.
alias(
    name = "framework",
    actual = "framework_experimental",
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
)

# The public target for the C++ API excluding experimental features.
cc_library(
    name = "framework_stable",
    srcs = [],
    hdrs = FRAMEWORK_STABLE_LIB_HDRS + CC_API_HDRS,
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:arena_planner",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:framework_lib",
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

# The public target for the full C++ API, including experimental features.
#
# Experimental APIs are functional, tested and usable in production; however,
# the corresponding API surface has not been finalized, and is subject to
# change.
#
# Note that if you have code which depends on both stable and experimental API
# features, it's fine to depend only on 'framework_experimental', since
# that includes 'framework_stable' as a subset.
cc_library(
    name = "framework_experimental",
    srcs = [],
    hdrs = FRAMEWORK_LIB_HDRS + CC_API_HDRS,
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
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

# DEPRECATED: use 'cc_api_stable' or 'cc_api_experimental' instead.
alias(
    name = "cc_api",
    actual = "cc_api_experimental",
    visibility = ["//tensorflow/lite:__pkg__"],
)

# The key parts of C++ API, excluding experimental features.
cc_library(
    name = "cc_api_stable",
    srcs = [],
    hdrs = CC_API_HDRS,
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//tensorflow/lite:__pkg__"],
    deps = [
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:arena_planner",
        "//tensorflow/lite:cc_api_stable",
        "//tensorflow/lite:external_cpu_backend_context",
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

# The key parts of C++ API, including experimental features.
cc_library(
    name = "cc_api_experimental",
    srcs = [],
    hdrs = CC_API_HDRS,
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//tensorflow/lite:__pkg__"],
    deps = [
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:arena_planner",
        "//tensorflow/lite:cc_api_experimental",
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
        "cc/kernels/builtin_op_kernels.h",
        "cc/kernels/register.h",
        # TODO(b/161243354): remove the line below when clients no
        # longer depend on it.
        "//tensorflow/lite/kernels:builtin_op_kernels.h",
        "//tensorflow/lite/kernels:fully_connected.h",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        ":builtin_ops_list",
        "//tensorflow/lite:cc_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)

# This target only _declares_ a method to create an op resolver, but does not define it. This allows
# clients to provide a custom implementation of the method, in particular to build in only a subset
# of supported ops. Providing any other custom implementation is done at clients' own risk.
cc_library(
    name = "create_op_resolver_header",
    srcs = [
        "//tensorflow/lite:create_op_resolver.h",
    ],
    hdrs = [
        "cc/create_op_resolver.h",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:op_resolver",
    ],
)

cc_library(
    name = "create_op_resolver_with_builtin_ops",
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:create_op_resolver_with_builtin_ops",
    ],
)

build_test(
    name = "cc_api_build_test",
    targets = [
        ":builtin_ops",
        ":create_op_resolver_header",
        ":create_op_resolver_with_builtin_ops",
        ":cc_api",
        ":cc_api_stable",
        ":cc_api_experimental",
        ":framework",
        ":framework_stable",
        ":framework_experimental",
    ],
)

cc_library(
    name = "delegate_registry",
    hdrs = ["cc/experimental/acceleration/configuration/delegate_registry.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
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
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/tools:verifier",
    ],
)

cc_library(
    name = "verifier_internal",
    hdrs = ["cc/tools/verifier_internal.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite/java/src/main/native:__pkg__",
    ],
    deps = [
        "//tensorflow/lite/tools:verifier_internal",
    ],
)

build_test(
    name = "verifier_build_test",
    targets = [
        ":verifier",
        ":verifier_internal",
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
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = ["//tensorflow/lite/c:c_api"],
)

cc_library(
    name = "c_api_without_op_resolver",
    hdrs = ["c/c_api.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = ["//tensorflow/lite/c:c_api_without_op_resolver"],
)

cc_library(
    name = "c_api_experimental",
    hdrs = ["c/c_api_experimental.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
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
        "@mediapipe//mediapipe/tasks:__subpackages__",
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
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = ["//tensorflow/lite/c:common"],
)

cc_library(
    name = "delegate_plugin",
    hdrs = ["c/experimental/acceleration/configuration/delegate_plugin.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/experimental/acceleration/configuration/c:delegate_plugin",
    ],
)

cc_library(
    name = "gpu_plugin",
    hdrs = ["c/experimental/acceleration/configuration/gpu_plugin.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/experimental/acceleration/configuration/c:gpu_plugin",
    ],
)

cc_library(
    name = "nnapi_plugin",
    hdrs = ["c/experimental/acceleration/configuration/nnapi_plugin.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/experimental/acceleration/configuration/c:nnapi_plugin",
    ],
)

cc_library(
    name = "xnnpack_plugin",
    hdrs = ["c/experimental/acceleration/configuration/xnnpack_plugin.h"],
    copts = tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/experimental/acceleration/configuration/c:xnnpack_plugin",
    ],
)

custom_c_library_with_tflite(
    name = "custom_c_api_example",
    testonly = True,
    models = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
    ],
    visibility = ["//visibility:private"],
)

build_test(
    name = "c_api_build_test",
    targets = [
        ":builtin_op_data",
        ":c_api",
        ":c_api_experimental",
        ":common",
        ":custom_c_api_example",
        ":delegate_plugin",
        ":nnapi_plugin",
        ":xnnpack_plugin",
    ],
)

cc_library(
    name = "builtin_ops_list",
    textual_hdrs = ["builtin_ops_list.inc"],
    visibility = ["//tensorflow/lite:__subpackages__"],
)

exports_files(
    srcs = ["builtin_ops_list.inc"],
    visibility = ["//tensorflow/lite:__subpackages__"],
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
        "@mediapipe//mediapipe/tasks:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support:__subpackages__",
    ],
)

cc_library(
    name = "cc_shims_test_util",
    testonly = True,
    hdrs = ["cc/shims_test_util.h"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "@mediapipe//mediapipe/tasks:__subpackages__",
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

cc_library(
    name = "jni_utils",
    srcs = ["jni/jni_utils.cc"],
    hdrs = ["jni/jni_utils.h"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/java/jni",
    ],
)

#------------------------------------------------------------------------------
# Bazel rules

bzl_library(
    name = "cc_library_with_tflite_bzl",
    srcs = ["cc_library_with_tflite.bzl"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite:build_def_bzl"],
)

#------------------------------------------------------------------------------

tflite_portable_test_suite()
