load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist", "tflite_portable_test_suite_combined")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

EMSCRIPTEN_LINKOPTS = [
    "-s ASSERTIONS=2",
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
    "-s EXIT_RUNTIME=1",
    "-s ALLOW_MEMORY_GROWTH=1",
    "-s TOTAL_MEMORY=134217728",
]

exports_files([
    "xnnpack_delegate.h",
])

# Force XNNPACK to run FP32 models as FP16, regardless of model metadata.
config_setting(
    name = "xnnpack_force_float_precision_explicit_fp16",
    define_values = {"xnnpack_force_float_precision": "fp16"},
)

# Force XNNPACK to use all operators in the delegate.
config_setting(
    name = "xnnpack_use_latest_ops_explicit",
    define_values = {"xnnpack_use_latest_ops": "true"},
)

# Force XNNPACK to enable subgraph reshaping.
config_setting(
    name = "xnnpack_enable_subgraph_reshaping_explicit",
    define_values = {"xnnpack_enable_subgraph_reshaping": "true"},
)

# Use transient indirection buffers.
config_setting(
    name = "xnnpack_use_transient_indirection_buffers_explicit",
    define_values = {"xnnpack_use_transient_indirection_buffers": "true"},
)

# Enable offloading of quantized 8-bit signed operators to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_qs8_explicit_true",
    define_values = {"tflite_with_xnnpack_qs8": "true"},
)

# Disable offloading of quantized 8-bit signed operators to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_qs8_explicit_false",
    define_values = {"tflite_with_xnnpack_qs8": "false"},
)

# Default setting for offloading of quantized 8-bit signed operators to XNNPACK delegate
cc_library(
    name = "tflite_with_xnnpack_qs8_default",
    compatible_with = get_compatible_with_portable(),
    defines = ["XNNPACK_DELEGATE_ENABLE_QS8=1"],
)

cc_library(
    name = "tflite_with_xnnpack_qs8",
    compatible_with = get_compatible_with_portable(),
    defines = select({
        ":tflite_with_xnnpack_qs8_explicit_true": ["XNNPACK_DELEGATE_ENABLE_QS8=1"],
        ":tflite_with_xnnpack_qs8_explicit_false": [],
        "//conditions:default": [],
    }),
    deps = select({
        ":tflite_with_xnnpack_qs8_explicit_true": [],
        ":tflite_with_xnnpack_qs8_explicit_false": [],
        "//conditions:default": [":tflite_with_xnnpack_qs8_default"],
    }),
)

# Enable offloading of quantized 8-bit unsigned operators to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_qu8_explicit_true",
    define_values = {"tflite_with_xnnpack_qu8": "true"},
)

# Disable offloading of quantized 8-bit unsigned operators to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_qu8_explicit_false",
    define_values = {"tflite_with_xnnpack_qu8": "false"},
)

# Default setting for offloading of quantized 8-bit unsigned operators to XNNPACK delegate
cc_library(
    name = "tflite_with_xnnpack_qu8_default",
    compatible_with = get_compatible_with_portable(),
    defines = ["XNNPACK_DELEGATE_ENABLE_QU8=1"],
)

cc_library(
    name = "tflite_with_xnnpack_qu8",
    compatible_with = get_compatible_with_portable(),
    defines = select({
        ":tflite_with_xnnpack_qu8_explicit_true": ["XNNPACK_DELEGATE_ENABLE_QU8=1"],
        ":tflite_with_xnnpack_qu8_explicit_false": [],
        "//conditions:default": [],
    }),
    deps = select({
        ":tflite_with_xnnpack_qu8_explicit_true": [],
        ":tflite_with_xnnpack_qu8_explicit_false": [],
        "//conditions:default": [":tflite_with_xnnpack_qu8_default"],
    }),
)

# Enable offloading of Fully Connected operator with dynamic weights to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_dynamic_fully_connected_explicit_true",
    define_values = {"tflite_with_xnnpack_dynamic_fully_connected": "true"},
)

# Disable offloading of Fully Connected operator with dynamic weights to XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_dynamic_fully_connected_explicit_false",
    define_values = {"tflite_with_xnnpack_dynamic_fully_connected": "false"},
)

# Default setting for offloading of Fully Connected operator with dynamic weights to XNNPACK delegate
cc_library(
    name = "tflite_with_xnnpack_dynamic_fully_connected_default",
    compatible_with = get_compatible_with_portable(),
    defines = [],
)

cc_library(
    name = "tflite_with_xnnpack_dynamic_fully_connected",
    compatible_with = get_compatible_with_portable(),
    defines = select({
        ":tflite_with_xnnpack_dynamic_fully_connected_explicit_true": ["XNNPACK_DELEGATE_ENABLE_DYNAMIC_FULLY_CONNECTED=1"],
        ":tflite_with_xnnpack_dynamic_fully_connected_explicit_false": [],
        "//conditions:default": [],
    }),
    deps = select({
        ":tflite_with_xnnpack_dynamic_fully_connected_explicit_true": [],
        ":tflite_with_xnnpack_dynamic_fully_connected_explicit_false": [],
        "//conditions:default": [":tflite_with_xnnpack_dynamic_fully_connected_default"],
    }),
)

# Enable logging of delegation errors in XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_logging_explicit_true",
    define_values = {"tflite_with_xnnpack_logging": "true"},
)

# Disable logging of delegation errors in XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_logging_explicit_false",
    define_values = {"tflite_with_xnnpack_logging": "false"},
)

# Default setting for logging of delegation errors in XNNPACK delegate
cc_library(
    name = "tflite_with_xnnpack_logging_default",
    compatible_with = get_compatible_with_portable(),
    defines = [],
)

cc_library(
    name = "tflite_with_xnnpack_logging",
    compatible_with = get_compatible_with_portable(),
    defines = select({
        ":tflite_with_xnnpack_logging_explicit_true": ["XNNPACK_DELEGATE_ENABLE_LOGGING=1"],
        ":tflite_with_xnnpack_logging_explicit_false": [],
        "//conditions:default": [],
    }),
    deps = select({
        ":tflite_with_xnnpack_logging_explicit_true": [],
        ":tflite_with_xnnpack_logging_explicit_false": [],
        "//conditions:default": [":tflite_with_xnnpack_logging_default"],
    }),
)

# Enable transient indirection buffers in XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_transient_indirection_buffer_explicit_true",
    define_values = {"tflite_with_xnnpack_transient_indirection_buffer": "true"},
)

# Disable transient indirection buffers in XNNPACK delegate
config_setting(
    name = "tflite_with_xnnpack_transient_indirection_buffer_explicit_false",
    define_values = {"tflite_with_xnnpack_transient_indirection_buffer": "false"},
)

# Default setting for transient indirection buffers in XNNPACK delegate
cc_library(
    name = "tflite_with_xnnpack_transient_indirection_buffer_default",
    compatible_with = get_compatible_with_portable(),
    defines = [],
)

cc_library(
    name = "tflite_with_xnnpack_transient_indirection_buffer",
    compatible_with = get_compatible_with_portable(),
    defines = select({
        ":tflite_with_xnnpack_transient_indirection_buffer_explicit_true": ["XNNPACK_DELEGATE_ENABLE_TRANSIENT_INDIRECTION_BUFFER=1"],
        ":tflite_with_xnnpack_transient_indirection_buffer_explicit_false": [],
        "//conditions:default": [],
    }),
    deps = select({
        ":tflite_with_xnnpack_transient_indirection_buffer_explicit_true": [],
        ":tflite_with_xnnpack_transient_indirection_buffer_explicit_false": [],
        "//conditions:default": [":tflite_with_xnnpack_transient_indirection_buffer_default"],
    }),
)

cc_library(
    name = "xnnpack_delegate",
    srcs = ["xnnpack_delegate.cc"],
    hdrs = ["xnnpack_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + select({
        ":xnnpack_force_float_precision_explicit_fp16": ["-DXNNPACK_DELEGATE_FORCE_PRECISION_FP16=1"],
        "//conditions:default": [],
    }) + select({
        ":xnnpack_enable_subgraph_reshaping_explicit": ["-DXNNPACK_DELEGATE_ENABLE_SUBGRAPH_RESHAPING=1"],
        # copybara:uncomment "//conditions:default": [],
        "//conditions:default": ["-DXNNPACK_DELEGATE_ENABLE_SUBGRAPH_RESHAPING=1"],  # copybara:comment
    }) + select({
        ":xnnpack_use_latest_ops_explicit": ["-DXNNPACK_DELEGATE_USE_LATEST_OPS=1"],
        # copybara:uncomment "//conditions:default": [],
        "//conditions:default": ["-DXNNPACK_DELEGATE_USE_LATEST_OPS=1"],  # copybara:comment
    }) + select({
        ":xnnpack_use_transient_indirection_buffers_explicit": ["-DXNNPACK_DELEGATE_USE_TRANSIENT_INDIRECTION_BUFFERS=1"],
        "//conditions:default": [],
    }) + ["-DFLATBUFFERS_LOCALE_INDEPENDENT=0"],
    linkstatic = True,
    deps = [
        ":file_util",
        ":flexbuffers_util",
        ":quantization_util",
        ":tflite_with_xnnpack_dynamic_fully_connected",  # buildcleaner: keep
        ":tflite_with_xnnpack_logging",  # buildcleaner: keep
        ":tflite_with_xnnpack_qs8",  # buildcleaner: keep
        ":tflite_with_xnnpack_qu8",  # buildcleaner: keep
        ":tflite_with_xnnpack_transient_indirection_buffer",  # buildcleaner: keep
        ":weight_cache",
        "//tensorflow/compiler/mlir/lite/kernels/internal:compatibility_macros",
        "//tensorflow/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "//tensorflow/lite:array",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@XNNPACK",
        "@eigen_archive//:eigen3",
        "@flatbuffers//:runtime_cc",
        "@pthreadpool",
    ],
)

cc_library(
    name = "xnnpack_delegate_hdrs_only",
    hdrs = ["xnnpack_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "xnnpack_delegate_test_mode",
    srcs = ["xnnpack_delegate.cc"],
    hdrs = ["xnnpack_delegate.h"],
    copts = tflite_copts() + ["-DXNNPACK_DELEGATE_TEST_MODE=1"] + ["-DFLATBUFFERS_LOCALE_INDEPENDENT=0"],
    linkstatic = True,
    deps = [
        ":file_util",
        ":flexbuffers_util",
        ":quantization_util",
        ":weight_cache",
        "//tensorflow/compiler/mlir/lite/kernels/internal:compatibility_macros",
        "//tensorflow/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "//tensorflow/lite:array",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@XNNPACK",
        "@eigen_archive//:eigen3",
        "@flatbuffers//:runtime_cc",
        "@pthreadpool",
    ],
)

cc_library(
    name = "quantization_util",
    srcs = ["quantization_util.cc"],
    hdrs = ["quantization_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:types",
        "@FP16",
    ],
)

flatbuffer_cc_library(
    name = "weight_cache_schema",
    srcs = ["weight_cache_schema.fbs"],
    compatible_with = get_compatible_with_portable(),
    flatc_args = [
        "--gen-mutable",
        "--gen-object-api",
    ],
)

cc_library(
    name = "weight_cache",
    srcs = ["weight_cache.cc"],
    hdrs = ["weight_cache.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":file_util",
        ":mmap_handle",
        ":weight_cache_schema",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "@XNNPACK",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "flexbuffers_util",
    hdrs = ["flexbuffers_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "weight_cache_test_helpers",
    testonly = True,
    hdrs = ["weight_cache_test_helpers.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":file_util",
        "//tensorflow/lite:minimal_logging",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mmap_handle",
    srcs = ["mmap_handle.cc"],
    hdrs = ["mmap_handle.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":file_util",
        "//tensorflow/lite:minimal_logging",
    ],
)

cc_test(
    name = "mmap_handle_test",
    srcs = ["mmap_handle_test.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":mmap_handle",
        ":weight_cache_test_helpers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "file_util",
    srcs = ["file_util.cc"],
    hdrs = ["file_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:minimal_logging",
    ],
)

################################ Tester classes ################################

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "binary_elementwise_tester",
    testonly = 1,
    srcs = ["binary_elementwise_tester.cc"],
    hdrs = ["binary_elementwise_tester.h"],
    deps = [
        ":test_util",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "batch_matrix_multiply_tester",
    testonly = 1,
    srcs = ["batch_matrix_multiply_tester.cc"],
    hdrs = ["batch_matrix_multiply_tester.h"],
    deps = [
        ":test_util",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "concatenation_tester",
    testonly = 1,
    srcs = ["concatenation_tester.cc"],
    hdrs = ["concatenation_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "conv_2d_tester",
    testonly = 1,
    srcs = ["conv_2d_tester.cc"],
    hdrs = ["conv_2d_tester.h"],
    deps = [
        ":test_util",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "depth_to_space_tester",
    testonly = 1,
    srcs = ["depth_to_space_tester.cc"],
    hdrs = ["depth_to_space_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "depthwise_conv_2d_tester",
    testonly = 1,
    srcs = ["depthwise_conv_2d_tester.cc"],
    hdrs = ["depthwise_conv_2d_tester.h"],
    deps = [
        ":test_util",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "dequantize_tester",
    testonly = 1,
    srcs = ["dequantize_tester.cc"],
    hdrs = ["dequantize_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "dynamically_quantized_fully_connected_tester",
    testonly = 1,
    srcs = ["dynamically_quantized_fully_connected_tester.cc"],
    hdrs = ["dynamically_quantized_fully_connected_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "dynamically_quantized_conv_2d_tester",
    testonly = 1,
    srcs = ["dynamically_quantized_conv_2d_tester.cc"],
    hdrs = ["dynamically_quantized_conv_2d_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "dynamically_quantized_transpose_conv_tester",
    testonly = 1,
    srcs = ["dynamically_quantized_transpose_conv_tester.cc"],
    hdrs = ["dynamically_quantized_transpose_conv_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "fully_connected_tester",
    testonly = 1,
    srcs = ["fully_connected_tester.cc"],
    hdrs = ["fully_connected_tester.h"],
    deps = [
        ":test_util",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "leaky_relu_tester",
    testonly = 1,
    srcs = ["leaky_relu_tester.cc"],
    hdrs = ["leaky_relu_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "odml_sdpa_tester",
    testonly = 1,
    srcs = ["odml_sdpa_tester.cc"],
    hdrs = ["odml_sdpa_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/experimental/genai:genai_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "pad_tester",
    testonly = 1,
    srcs = ["pad_tester.cc"],
    hdrs = ["pad_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "pool_2d_tester",
    testonly = 1,
    srcs = ["pool_2d_tester.cc"],
    hdrs = ["pool_2d_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "prelu_tester",
    testonly = 1,
    srcs = ["prelu_tester.cc"],
    hdrs = ["prelu_tester.h"],
    deps = [
        ":test_util",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantize_tester",
    testonly = 1,
    srcs = ["quantize_tester.cc"],
    hdrs = ["quantize_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_binary_elementwise_tester",
    testonly = 1,
    srcs = ["quantized_binary_elementwise_tester.cc"],
    hdrs = ["quantized_binary_elementwise_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_conv_2d_tester",
    testonly = 1,
    srcs = ["quantized_conv_2d_tester.cc"],
    hdrs = ["quantized_conv_2d_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_depthwise_conv_2d_tester",
    testonly = 1,
    srcs = ["quantized_depthwise_conv_2d_tester.cc"],
    hdrs = ["quantized_depthwise_conv_2d_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_fully_connected_tester",
    testonly = 1,
    srcs = ["quantized_fully_connected_tester.cc"],
    hdrs = ["quantized_fully_connected_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_leaky_relu_tester",
    testonly = 1,
    srcs = ["quantized_leaky_relu_tester.cc"],
    hdrs = ["quantized_leaky_relu_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_pad_tester",
    testonly = 1,
    srcs = ["quantized_pad_tester.cc"],
    hdrs = ["quantized_pad_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_pool_2d_tester",
    testonly = 1,
    srcs = ["quantized_pool_2d_tester.cc"],
    hdrs = ["quantized_pool_2d_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_resize_bilinear_tester",
    testonly = 1,
    srcs = ["quantized_resize_bilinear_tester.cc"],
    hdrs = ["quantized_resize_bilinear_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_unary_elementwise_tester",
    testonly = 1,
    srcs = ["quantized_unary_elementwise_tester.cc"],
    hdrs = ["quantized_unary_elementwise_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_variable_ops_tester",
    testonly = 1,
    srcs = ["quantized_variable_ops_tester.cc"],
    hdrs = ["quantized_variable_ops_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

# Same as quantized_variable_ops_tester above, but depends on xnnpack_delegate (no test
# mode), used for quantized_variable_ops_multiple_subgraph_test. Why name this
# no_test_mode instead of renaming the above to test_mode, because all testers
# depend on test_mode by default, this keeps the name consistent.
cc_library(
    name = "quantized_variable_ops_tester_no_test_mode",
    testonly = 1,
    srcs = ["quantized_variable_ops_tester.cc"],
    hdrs = ["quantized_variable_ops_tester.h"],
    deps = [
        ":xnnpack_delegate",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "reduce_tester",
    testonly = 1,
    srcs = ["reduce_tester.cc"],
    hdrs = ["reduce_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "reshape_tester",
    testonly = 1,
    srcs = ["reshape_tester.cc"],
    hdrs = ["reshape_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "resize_bilinear_tester",
    testonly = 1,
    srcs = ["resize_bilinear_tester.cc"],
    hdrs = ["resize_bilinear_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "slice_tester",
    testonly = 1,
    srcs = ["slice_tester.cc"],
    hdrs = ["slice_tester.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "split_tester",
    testonly = 1,
    srcs = ["split_tester.cc"],
    hdrs = ["split_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "softmax_tester",
    testonly = 1,
    srcs = ["softmax_tester.cc"],
    hdrs = ["softmax_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "space_to_depth_tester",
    testonly = 1,
    srcs = ["space_to_depth_tester.cc"],
    hdrs = ["space_to_depth_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "strided_slice_tester",
    testonly = 1,
    srcs = ["strided_slice_tester.cc"],
    hdrs = ["strided_slice_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "unary_elementwise_tester",
    testonly = 1,
    srcs = ["unary_elementwise_tester.cc"],
    hdrs = ["unary_elementwise_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "transpose_tester",
    testonly = 1,
    srcs = ["transpose_tester.cc"],
    hdrs = ["transpose_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "transpose_conv_tester",
    testonly = 1,
    srcs = ["transpose_conv_tester.cc"],
    hdrs = ["transpose_conv_tester.h"],
    deps = [
        ":test_util",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantized_transpose_conv_tester",
    testonly = 1,
    srcs = ["quantized_transpose_conv_tester.cc"],
    hdrs = ["quantized_transpose_conv_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/compiler/mlir/lite/schema:schema_conversion_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "variable_ops_tester",
    testonly = 1,
    srcs = ["variable_ops_tester.cc"],
    hdrs = ["variable_ops_tester.h"],
    deps = [
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

# Same as variable_ops_tester above, but depends on xnnpack_delegate (no test
# mode), used for variable_ops_multiple_subgraph_test. Why name this
# no_test_mode instead of renaming the above to test_mode, because all testers
# depend on test_mode by default, this keeps the name consistent.
cc_library(
    name = "variable_ops_tester_no_test_mode",
    testonly = 1,
    srcs = ["variable_ops_tester.cc"],
    hdrs = ["variable_ops_tester.h"],
    deps = [
        ":xnnpack_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

############################## Integration tests ###############################

cc_library(
    name = "test_main",
    testonly = 1,
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "average_pool_2d_test",
    srcs = ["average_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "batch_matrix_multiply_test",
    srcs = ["batch_matrix_multiply_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":batch_matrix_multiply_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "binary_elementwise_test",
    srcs = [
        "binary_elementwise_test.cc",
    ],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "channelwise_quantized_conv_2d_test",
    srcs = ["channelwise_quantized_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "channelwise_quantized_depthwise_conv_2d_test",
    srcs = ["channelwise_quantized_depthwise_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_depthwise_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "concatenation_test",
    srcs = ["concatenation_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":concatenation_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "conv_2d_test",
    srcs = ["conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "delegate_test",
    srcs = ["delegate_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
        "@pthreadpool",
    ],
)

cc_test(
    name = "depth_to_space_test",
    srcs = ["depth_to_space_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":depth_to_space_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "depthwise_conv_2d_test",
    srcs = ["depthwise_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":depthwise_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dynamically_quantized_fully_connected_test",
    srcs = ["dynamically_quantized_fully_connected_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    # TODO(b/360983009): This test are currently failing on TAP presubmit for
    # android x86. We should re-enable it once we fix the failures.
    tags = ["notap"],
    deps = [
        ":dynamically_quantized_fully_connected_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dynamically_quantized_conv_2d_test",
    srcs = ["dynamically_quantized_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":dynamically_quantized_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dynamically_quantized_transpose_conv_test",
    srcs = ["dynamically_quantized_transpose_conv_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":dynamically_quantized_transpose_conv_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":fully_connected_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "leaky_relu_test",
    srcs = ["leaky_relu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":leaky_relu_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "max_pool_2d_test",
    srcs = ["max_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "reduce_test",
    srcs = ["reduce_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":reduce_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "odml_sdpa_test",
    srcs = ["odml_sdpa_test.cc"],
    data = [
        ":odml_sdpa_composite_gqa.tflite.bin",
        ":odml_sdpa_composite_mha.tflite.bin",
        ":odml_sdpa_composite_mqa.tflite.bin",
    ],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":odml_sdpa_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "pad_test",
    srcs = ["pad_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":pad_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "prelu_test",
    srcs = ["prelu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":prelu_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantization_util_test",
    srcs = ["quantization_util_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantization_util",
        ":test_main",
        "//tensorflow/lite/kernels/internal:types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantize_float32_to_int8_test",
    srcs = ["quantize_float32_to_int8_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantize_float32_to_uint8_test",
    srcs = ["quantize_float32_to_uint8_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantize_int8_to_int8_test",
    srcs = ["quantize_int8_to_int8_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantize_uint8_to_uint8_test",
    srcs = ["quantize_uint8_to_uint8_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "variable_ops_multiple_subgraph_test",
    srcs = ["variable_ops_multiple_subgraph_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        # Don't use xnnpack_delegate_test_mode because we use CALL_ONCE here,
        # which XNNPACK does not yet support, and test mode enforces that there
        # are no unsupported operations.
        ":xnnpack_delegate",
        ":variable_ops_tester_no_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "variable_ops_test",
    srcs = ["variable_ops_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":variable_ops_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "reshape_test",
    srcs = ["reshape_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":reshape_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@XNNPACK",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_reshape_test",
    srcs = ["signed_quantized_reshape_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":reshape_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_reshape_test",
    srcs = ["unsigned_quantized_reshape_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":reshape_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "resize_bilinear_test",
    srcs = ["resize_bilinear_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":resize_bilinear_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_dequantize_test",
    srcs = ["signed_dequantize_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":dequantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_add_test",
    srcs = ["signed_quantized_add_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_concatenation_test",
    srcs = ["signed_quantized_concatenation_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":concatenation_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_conv_2d_test",
    srcs = ["signed_quantized_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_depthwise_conv_2d_test",
    srcs = ["signed_quantized_depthwise_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_depthwise_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_depth_to_space_test",
    srcs = ["signed_quantized_depth_to_space_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":depth_to_space_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_elu_test",
    srcs = ["signed_quantized_elu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_unary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_fully_connected_test",
    srcs = ["signed_quantized_fully_connected_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_fully_connected_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_leaky_relu_test",
    srcs = ["signed_quantized_leaky_relu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_leaky_relu_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_logistic_test",
    srcs = ["signed_quantized_logistic_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_unary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_max_pool_2d_test",
    srcs = ["signed_quantized_max_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_mul_test",
    srcs = ["signed_quantized_mul_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_pad_test",
    srcs = ["signed_quantized_pad_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_pad_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_resize_bilinear_test",
    srcs = ["signed_quantized_resize_bilinear_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_resize_bilinear_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_slice_test",
    srcs = ["signed_quantized_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":slice_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_space_to_depth_test",
    srcs = ["signed_quantized_space_to_depth_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":space_to_depth_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_split_test",
    srcs = ["signed_quantized_split_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":split_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_strided_slice_test",
    srcs = ["signed_quantized_strided_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":strided_slice_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_sub_test",
    srcs = ["signed_quantized_sub_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_tanh_test",
    srcs = ["signed_quantized_tanh_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_unary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_transpose_test",
    srcs = ["signed_quantized_transpose_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":transpose_tester",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_transpose_conv_test",
    srcs = ["signed_quantized_transpose_conv_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_transpose_conv_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "signed_quantized_variable_ops_multiple_subgraph_test",
    srcs = ["signed_quantized_variable_ops_multiple_subgraph_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        # Don't use xnnpack_delegate_test_mode because we use CALL_ONCE here,
        # which XNNPACK does not yet support, and test mode enforces that there
        # are no unsupported operations.
        ":xnnpack_delegate",
        ":quantized_variable_ops_tester_no_test_mode",
        "@com_google_googletest//:gtest",
        "//tensorflow/lite/c:c_api_types",
    ],
)

cc_test(
    name = "signed_quantized_variable_ops_test",
    srcs = ["signed_quantized_variable_ops_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_variable_ops_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "slice_test",
    srcs = ["slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":slice_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "softmax_test",
    srcs = ["softmax_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":softmax_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:c_api_types",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "space_to_depth_test",
    srcs = ["space_to_depth_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":space_to_depth_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "split_test",
    srcs = ["split_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":split_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "strided_slice_test",
    srcs = ["strided_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":strided_slice_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":transpose_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "transpose_conv_test",
    srcs = ["transpose_conv_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":transpose_conv_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unary_elementwise_test",
    srcs = ["unary_elementwise_test.cc"],
    deps = [
        ":unary_elementwise_tester",
        ":xnnpack_delegate_hdrs_only",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "unsigned_dequantize_test",
    srcs = ["unsigned_dequantize_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":dequantize_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_add_test",
    srcs = ["unsigned_quantized_add_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_concatenation_test",
    srcs = ["unsigned_quantized_concatenation_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":concatenation_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_conv_2d_test",
    srcs = ["unsigned_quantized_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_depthwise_conv_2d_test",
    srcs = ["unsigned_quantized_depthwise_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_depthwise_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_depth_to_space_test",
    srcs = ["unsigned_quantized_depth_to_space_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":depth_to_space_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_fully_connected_test",
    srcs = ["unsigned_quantized_fully_connected_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_fully_connected_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_leaky_relu_test",
    srcs = ["unsigned_quantized_leaky_relu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_leaky_relu_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_logistic_test",
    srcs = ["unsigned_quantized_logistic_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_unary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_max_pool_2d_test",
    srcs = ["unsigned_quantized_max_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_mul_test",
    srcs = ["unsigned_quantized_mul_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_pad_test",
    srcs = ["unsigned_quantized_pad_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_pad_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_resize_bilinear_test",
    srcs = ["unsigned_quantized_resize_bilinear_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_resize_bilinear_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_slice_test",
    srcs = ["unsigned_quantized_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":slice_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_space_to_depth_test",
    srcs = ["unsigned_quantized_space_to_depth_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":space_to_depth_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_split_test",
    srcs = ["unsigned_quantized_split_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":split_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_strided_slice_test",
    srcs = ["unsigned_quantized_strided_slice_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":strided_slice_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_sub_test",
    srcs = ["unsigned_quantized_sub_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_tanh_test",
    srcs = ["unsigned_quantized_tanh_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_unary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_transpose_test",
    srcs = ["unsigned_quantized_transpose_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":transpose_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_transpose_conv_test",
    srcs = ["unsigned_quantized_transpose_conv_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_transpose_conv_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_variable_ops_multiple_subgraph_test",
    srcs = ["unsigned_quantized_variable_ops_multiple_subgraph_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        # Don't use xnnpack_delegate_test_mode because we use CALL_ONCE here,
        # which XNNPACK does not yet support, and test mode enforces that there
        # are no unsupported operations.
        ":xnnpack_delegate",
        ":quantized_variable_ops_tester_no_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unsigned_quantized_variable_ops_test",
    srcs = ["unsigned_quantized_variable_ops_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":quantized_variable_ops_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "weights_cache_test",
    srcs = ["weights_cache_test.cc"],
    deps = [
        ":conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/core:framework",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "weight_cache_test",
    srcs = ["weight_cache_test.cc"],
    deps = [
        ":file_util",
        ":mmap_handle",
        ":test_main",
        ":weight_cache",
        ":weight_cache_schema",
        ":weight_cache_test_helpers",
        ":xnnpack_delegate_test_mode",
        "//tensorflow/lite/c:common",
        "@XNNPACK",
        "@com_google_googletest//:gtest",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "delegate_weight_cache_test",
    srcs = ["delegate_weight_cache_test.cc"],
    deps = [
        ":weight_cache",
        ":xnnpack_delegate_hdrs_only",
        "//tensorflow/lite:framework_experimental",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:model_building",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "flexbuffers_util_test",
    srcs = ["flexbuffers_util_test.cc"],
    deps = [
        ":flexbuffers_util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "file_util_test",
    srcs = ["file_util_test.cc"],
    deps = [
        ":file_util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

tflite_portable_test_suite_combined(combine_conditions = {"deps": [":test_main"]})
