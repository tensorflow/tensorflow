load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_self_contained_libs_test_suite")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist", "tflite_portable_test_suite")
load("//tensorflow/lite/core:special_rules.bzl", "core_cc_api_stable_visibility_allowlist", "macros_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "create_op_resolver.h",
        "macros.h",
        "subgraph.h",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
)

bzl_library(
    name = "special_rules_bzl",
    srcs = ["special_rules.bzl"],
    visibility = ["//tensorflow/lite:__subpackages__"],
)

# The public target for the C++ API excluding experimental APIs.
# TODO(ahentz): investigate dependency on gemm_support requiring usage of tf_copts.
cc_library(
    name = "framework_stable",
    srcs = [
        "subgraph.h",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "macros.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = [
        ":cc_api_stable",
        ":signature_runner",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/compiler/mlir/lite/experimental/remat:metadata_util",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:array",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:graph_info",
        "//tensorflow/lite:interpreter_options_header",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:memory_planner",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite:string",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/async:async_signature_runner",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/profiling:root_profiler",
        "//tensorflow/lite/profiling/telemetry:profiler",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting_internal",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

# The public target for the full C++ API, including experimental APIs.
#
# Experimental APIs are functional, tested and usable in production; however,
# the corresponding API surface has not been finalized, and is subject to
# change.
alias(
    name = "framework",
    actual = "framework_experimental",
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
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
cc_library(
    name = "framework_experimental",
    srcs = [],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "macros.h",
        "model.h",
        "model_builder.h",
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":cc_api_experimental",
        ":cc_api_stable",
        ":model_builder",
        ":signature_runner",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/compiler/mlir/lite/experimental/remat:metadata_util",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:array",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:graph_info",
        "//tensorflow/lite:interpreter_options_header",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:memory_planner",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite:string",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/async:async_signature_runner",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/profiling:root_profiler",
        "//tensorflow/lite/profiling/telemetry:profiler",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting_internal",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

# This is a private target, its visibility is set to public only to be
# used by LiteRT dependencies.
# Do not use this target directly and don't consider it as a part of the public API.
# TODO(weiyiw): Refactor LiteRT deps from TFLite.
alias(
    name = "private_cc_api_stable",
    actual = ":cc_api_stable",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

# TODO(b/242310498): move logger.cc from tensorflow/lite/ to here.
cc_library(
    name = "cc_api_stable",
    srcs = [
        "interpreter.cc",
        "interpreter_builder.cc",
        "subgraph.h",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "//third_party/odml/litert/litert:__subpackages__",
    ] + core_cc_api_stable_visibility_allowlist(),
    deps = [
        ":model_builder",
        ":signature_runner",
        ":subgraph",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/compiler/mlir/lite/experimental/remat:metadata_util",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/schema:schema_utils",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:array",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:graph_info",
        "//tensorflow/lite:interpreter_options_header",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:memory_planner",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:shared_library",
        "//tensorflow/lite:simple_memory_arena",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite:string",
        "//tensorflow/lite:tensorflow_profiler_logger_shim",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite:util",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:verifier",
        "//tensorflow/lite/core/async:async_signature_runner",
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
        "//tensorflow/lite/profiling/telemetry",
        "//tensorflow/lite/profiling/telemetry:profiler",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting_internal",
        "//tensorflow/lite/schema:conversion_metadata_fbs",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
        "@ruy//ruy:denormal",
    ],
    alwayslink = 1,
)

# The key parts of the C++ API.  This target defines the TF Lite classes for
# loading models and interpreting them.
# DEPRECATED: prefer to depend on :cc_api_stable or :cc_api_experimental.
alias(
    name = "cc_api",
    actual = "cc_api_experimental",
    visibility = ["//tensorflow/lite:__subpackages__"],
)

# The key parts of the C++ API, including experimental APIs.
#
# This target has restricted visibility; for a public target that exposes
# these APIs, see 'framework_experimental' above.
cc_library(
    name = "cc_api_experimental",
    srcs = [
        "interpreter_experimental.cc",
    ],
    hdrs = [
        "interpreter.h",
        "interpreter_builder.h",
        "model.h",
        "model_builder.h",
        "signature_runner.h",
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":cc_api_stable",
        ":signature_runner",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/compiler/mlir/lite/experimental/remat:metadata_util",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:array",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:graph_info",
        "//tensorflow/lite:interpreter_options_header",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:memory_planner",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite:string",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/async:async_signature_runner",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/profiling:root_profiler",
        "//tensorflow/lite/profiling/telemetry:profiler",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting",
        "//tensorflow/lite/profiling/telemetry/c:telemetry_setting_internal",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

cc_library(
    name = "model_builder",
    hdrs = ["model_builder.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = internal_visibility_allowlist(),
    deps = [
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite/core/api:error_reporter",
    ],
    alwayslink = 1,
)

cc_library(
    name = "signature_runner",
    srcs = ["signature_runner.cc"],
    hdrs = ["signature_runner.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__pkg__",
        "//tensorflow/lite/core:__subpackages__",
        "//third_party/odml/infra/genai/inference/executor/google_tensor:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/internal:signature_def",
    ],
)

# Test signature runner.
cc_test(
    name = "signature_runner_test",
    size = "small",
    srcs = ["signature_runner_test.cc"],
    data = [
        "//tensorflow/lite:testdata/multi_signatures.bin",
        "//tensorflow/lite:testdata/reverse_signature_model.bin",
    ],
    deps = [
        ":framework",
        ":signature_runner",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Test model framework.
cc_test(
    name = "model_test",
    size = "small",
    srcs = ["model_test.cc"],
    data = [
        "//tensorflow/lite:testdata/0_subgraphs.bin",
        "//tensorflow/lite:testdata/2_subgraphs.bin",
        "//tensorflow/lite:testdata/2_subgraphs_dont_delegate_name.bin",
        "//tensorflow/lite:testdata/add_shared_tensors.bin",
        "//tensorflow/lite:testdata/empty_model.bin",
        "//tensorflow/lite:testdata/multi_add_flex.bin",
        "//tensorflow/lite:testdata/segment_sum_invalid_buffer.bin",
        "//tensorflow/lite:testdata/sparse_tensor.bin",
        "//tensorflow/lite:testdata/test_min_runtime.bin",
        "//tensorflow/lite:testdata/test_model.bin",
        "//tensorflow/lite:testdata/test_model_broken.bin",
        "//tensorflow/lite:testdata/test_model_redux_precision.bin",
        "//tensorflow/lite:testdata/while_op_with_forwarding_input.bin",
        "//tensorflow/lite:testdata/zero_size_constant.bin",
    ],
    tags = [
        "no_windows",  # TODO(b/194459105): the test is flaky.
        "noasan",
        "tflite_not_portable",
        "tflite_smoke_test",
    ],
    deps = [
        ":framework",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:verifier",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "create_op_resolver_header",
    hdrs = [
        "create_op_resolver.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:op_resolver",
    ],
)

# Defines CreateOpResolver with all builtin ops.
cc_library(
    name = "create_op_resolver_with_builtin_ops",
    srcs = ["create_op_resolver_with_builtin_ops.cc"],
    hdrs = ["create_op_resolver.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:op_resolver",
        "//tensorflow/lite/core/kernels:builtin_ops",
    ],
    # Some targets only have an implicit dependency on CreateOpResolver.
    # This avoids warnings about backwards references when linking.
    alwayslink = True,
)

# This target is only for use by the "tflite_custom_c_library" and "tflite_custom_cc_library" build
# macro and should not be used anywhere other than in the implementation of that build macro.
# "tflite_custom_c_library" requires target to be public, that's why we duplicated
# :create_op_resolver_header target to be used only by "tflite_custom_c_library".
# Making :create_op_resolver_header public could cause some problems because it is widely used
# inside the TF Lite code base, that might lead others outside the TF Lite code base to copy that
# dependency and use it and subsequently depend on it, which would be bad. Using a separate
# :private_create_op_resolver_header target ensures that the only use of the unwantedly-"public"
# target is inside the "tflite_custom_c_library" itself, where it is less likely to get copied into
# third party code.
alias(
    name = "private_create_op_resolver_header",
    actual = ":create_op_resolver_header",
    tags = ["avoid_dep"],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ] + macros_visibility_allowlist(),
)

cc_library(
    name = "subgraph",
    srcs = [
        "subgraph.cc",
    ],
    hdrs = [
        "subgraph.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        "//platforms/darwinn/tflite:__subpackages__",
        "//tensorflow/lite:__subpackages__",
        "//tensorflow/lite/core:__subpackages__",
        "//tensorflow/lite/kernels:__subpackages__",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/experimental/remat:metadata_util",
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:array",
        "//tensorflow/lite:graph_info",
        "//tensorflow/lite:interpreter_options_header",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:memory_planner",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common_internal",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/profiling:root_profiler",
        "//tensorflow/lite/profiling/telemetry",
        "//tensorflow/lite/schema:schema_fbs",
    ] + select({
        "//tensorflow/lite:tflite_use_simple_memory_planner": [
            "//tensorflow/lite:simple_planner",
        ],
        "//conditions:default": [
            "//tensorflow/lite:arena_planner",
        ],
    }) + select({
        "//tensorflow/lite:tensorflow_profiler_config": [
            "//tensorflow/lite:tensorflow_profiler_logger_shim",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # TODO(b/161243354): eliminate this.
)

# Test subgraph.
cc_test(
    name = "subgraph_test",
    size = "small",
    srcs = [
        "subgraph_test.cc",
    ],
    deps = [
        ":framework_stable",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/kernels:builtin_ops",  # build_cleaner: keep
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_building",
    srcs = ["model_building.cc"],
    hdrs = ["model_building.h"],
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":subgraph",
        "//tensorflow/lite:array",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test(
    name = "model_building_test",
    srcs = ["model_building_test.cc"],
    deps = [
        ":model_building",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "subgraph_composite_inlining_test",
    srcs = ["subgraph_composite_inlining_test.cc"],
    deps = [
        ":model_building",
        ":subgraph",
        "//tensorflow/lite:array",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:subgraph_test_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:tensor_ctypes",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")

tflite_portable_test_suite()
