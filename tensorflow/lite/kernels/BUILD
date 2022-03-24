load("//tensorflow/lite:build_def.bzl", "tflite_combine_cc_tests", "tflite_copts", "tflite_self_contained_libs_test_suite")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite_combined")
load("//tensorflow:tensorflow.bzl", "tf_opts_nortti_if_android")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable", "pybind_extension")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Enables usage of ruy exclusively as the GEMM backend in TFLite kernels.
# This will cause TFLite to build with ruy only, providing a smaller binary.
# WARNING: This build flag is experimental and subject to change.
config_setting(
    name = "tflite_with_ruy_explicit_true",
    define_values = {"tflite_with_ruy": "true"},
)

# Disables usage of ruy as the exclusive GEMM backend in TFLite kernels.
# TFLite will be built with ruy and other GEMM libraries. Ruy will not be
# the default GEMM option at runtime.
# WARNING: This build flag is experimental and subject to change.
config_setting(
    name = "tflite_with_ruy_explicit_false",
    define_values = {"tflite_with_ruy": "false"},
)

###### Beginning of config_setting's to match aarch64 ######
#
# We need to identify the aarch64 instruction set to decide whether to enable
# TFLITE_WITH_RUY by default. This is surprisingly hard to do because select()
# can only consume config_setting's, these config_settings are not centralized,
# and the "cpu" value which they define are free-form strings and there is no
# standardization of the strings that we need to match for the aarch64 architecture.
#
# First, we have the case of --config=chromiumos_arm, which defines cpu=arm but is
# actually aarch64. For it, we name our config_setting chromiumos_arm64 to avoid
# adding to the confusion, at the cost of diverging from the --config name.
# This example shows that we can never hope to match aarch64 by looking only at
# "cpu", since the value "arm" would be used to mean the (32-bit) ARM instruction set
# in other configs.
config_setting(
    name = "chromiumos_arm64",
    values = {
        "crosstool_top": "//external:chromiumos/crosstool",
        "cpu": "arm",
    },
    visibility = ["//visibility:private"],
)

# Next, several "cpu" values that unambiguously mean aarch64, that are observed in
# practice with --config's that we care to support:

# This is defined by the tensorflow:linux_aarch64 config_setting.
config_setting(
    name = "cpu_aarch64",
    values = {"cpu": "aarch64"},
    visibility = ["//visibility:private"],
)

# This is defined by some config_setting's in the wild and is a reasonable value to
# support anyway.
config_setting(
    name = "cpu_arm64",
    values = {"cpu": "arm64"},
    visibility = ["//visibility:private"],
)

# This is the value defined by --config=ios_arm64.
config_setting(
    name = "cpu_ios_arm64",
    values = {"cpu": "ios_arm64"},
    visibility = ["//visibility:private"],
)

# arm64e variants of the above two. See:
# https://stackoverflow.com/questions/52624308/xcode-arm64-vs-arm64e
config_setting(
    name = "cpu_arm64e",
    values = {"cpu": "arm64e"},
    visibility = ["//visibility:private"],
)

config_setting(
    name = "cpu_ios_arm64e",
    values = {"cpu": "ios_arm64e"},
    visibility = ["//visibility:private"],
)

# This is the value defined by --config=android_arm64
config_setting(
    name = "cpu_arm64_v8a",
    values = {"cpu": "arm64-v8a"},
    visibility = ["//visibility:private"],
)

###### End of config_setting's to match aarch64 ######

# Suppress warnings that are introduced by Eigen Tensor.
EXTRA_EIGEN_COPTS = select({
    "//tensorflow:ios": [
        "-Wno-error=invalid-partial-specialization",
        "-Wno-error=reorder",
    ],
    "//tensorflow:windows": [
        "/DEIGEN_HAS_C99_MATH",
        "/DEIGEN_AVOID_STL_ARRAY",
    ],
    "//conditions:default": ["-Wno-error=reorder"],
})

cc_test(
    name = "optional_tensor_test",
    size = "small",
    srcs = ["optional_tensor_test.cc"],
    deps = [
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "acceleration_test_util",
    testonly = 1,
    srcs = [
        "acceleration_test_util.cc",
    ],
    hdrs = ["acceleration_test_util.h"],
    deps = [
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "acceleration_test_util_internal",
    testonly = 1,
    srcs = [
        "acceleration_test_util_internal.cc",
    ],
    hdrs = ["acceleration_test_util_internal.h"],
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        "@com_google_absl//absl/types:optional",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "acceleration_test_util_internal_test",
    srcs = [
        "acceleration_test_util_internal_test.cc",
    ],
    deps = [
        ":acceleration_test_util_internal",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":acceleration_test_util",
        ":builtin_ops",
        ":test_delegate_providers_lib",
        "//tensorflow/core/platform:logging",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:simple_planner",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/delegates/nnapi:acceleration_test_util",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools/optimize:quantization_utils",
        "//tensorflow/lite/tools/versioning",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

# A convenient library of tflite delegate execution providers for kernel tests
# based on SingleOpModel or its derivatives defined in test_util.h/cc.
cc_library(
    name = "test_util_delegate_providers",
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/tools/delegates:coreml_delegate_provider",
        "//tensorflow/lite/tools/delegates:default_execution_provider",
        "//tensorflow/lite/tools/delegates:delegate_provider_lib",
        "//tensorflow/lite/tools/delegates:external_delegate_provider",
        "//tensorflow/lite/tools/delegates:hexagon_delegate_provider",
        "//tensorflow/lite/tools/delegates:nnapi_delegate_provider",
        "//tensorflow/lite/tools/delegates:xnnpack_delegate_provider",
    ] + select({
        # Metal GPU delegate for iOS has its own setups for kernel tests, so
        # skipping linking w/ the gpu_delegate_provider.
        "//tensorflow:ios": [],
        "//conditions:default": [
            "//tensorflow/lite/tools/delegates:gpu_delegate_provider",
        ],
    }),
    alwayslink = 1,
    # TODO(b/161243354): add testonly=1?
)

cc_library(
    name = "test_delegate_providers_lib",
    srcs = ["test_delegate_providers.cc"],
    hdrs = ["test_delegate_providers.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "//tensorflow/lite/tools/delegates:delegate_provider_lib",
    ],
)

# TODO(b/132204084): Create tflite_cc_test rule to automate test_main inclusion.
cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.cc"],
    deps = [
        ":test_delegate_providers_lib",
        ":test_util",
        ":test_util_delegate_providers",
        "//tensorflow/lite/testing:util",
        "//tensorflow/lite/tools:command_line_flags",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "eigen_support",
    srcs = [
        "eigen_support.cc",
    ],
    hdrs = [
        "eigen_support.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + EXTRA_EIGEN_COPTS,
    visibility = ["//visibility:private"],
    deps = [
        ":op_macros",
        "//tensorflow/lite:arena_planner",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:optimized_eigen",
    ],
)

cc_test(
    name = "eigen_support_test",
    size = "small",
    srcs = ["eigen_support_test.cc"],
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":eigen_support",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:optimized_eigen",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tflite_with_ruy_enabled",
    compatible_with = get_compatible_with_portable(),
    defines = ["TFLITE_WITH_RUY"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "tflite_with_ruy_default",
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = select({
        ":chromiumos_arm64": [":tflite_with_ruy_enabled"],
        ":cpu_aarch64": [":tflite_with_ruy_enabled"],
        ":cpu_arm64": [":tflite_with_ruy_enabled"],
        ":cpu_arm64e": [":tflite_with_ruy_enabled"],
        ":cpu_ios_arm64": [":tflite_with_ruy_enabled"],
        ":cpu_ios_arm64e": [":tflite_with_ruy_enabled"],
        ":cpu_arm64_v8a": [":tflite_with_ruy_enabled"],
        "//tensorflow:android_arm": ["tflite_with_ruy_enabled"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tflite_with_ruy",
    compatible_with = get_compatible_with_portable(),
    deps = select({
        ":tflite_with_ruy_explicit_true": [":tflite_with_ruy_enabled"],
        ":tflite_with_ruy_explicit_false": [],
        "//conditions:default": [":tflite_with_ruy_default"],
    }),
)

# Provide a library for clients to link to if they need to stay on deprecated
# arithmetic backends. Include as a dependency of cpu_backend_gemm to start.
# TODO(b/168923364): Move to dependent targets.
cc_library(
    name = "deprecated_backends",
    srcs = [
        "deprecated_backends.cc",
    ],
    compatible_with = get_compatible_with_portable(),
    alwayslink = 1,
)

cc_library(
    name = "cpu_backend_context",
    srcs = [
        "cpu_backend_context.cc",
    ],
    hdrs = [
        "cpu_backend_context.h",
    ],
    compatible_with = get_compatible_with_portable(),
    # TF Lite builds in other build systems should "opt in" to cpufinfo.
    copts = tflite_copts() + select({
        "//tensorflow:linux_ppc64le": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:fuchsia": [],
        "//conditions:default": ["-DTFLITE_HAVE_CPUINFO"],
    }),
    deps = [
        ":tflite_with_ruy",
        ":op_macros",
        # For now this unconditionally depends on both ruy and gemmlowp.
        # See the comment inside class CpuBackendContext on the
        # gemmlowp_context_ and ruy_context_ members.
        "@ruy//ruy:context",
        "@ruy//ruy:path",
        "@gemmlowp",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:macros",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite/kernels/internal:compatibility",
    ] + select({
        # This select must match the similar select in `copts`
        "//tensorflow:linux_ppc64le": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:fuchsia": [],
        "//conditions:default": ["@cpuinfo//:cpuinfo_with_unstripped_include_path"],
    }),
)

cc_library(
    name = "cpu_backend_threadpool",
    hdrs = [
        "cpu_backend_threadpool.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":cpu_backend_context",
        ":tflite_with_ruy",
        "//tensorflow/lite/kernels/internal:compatibility",
        # For now this unconditionally depends on both ruy and gemmlowp.
        # We only need to depend on gemmlowp when tflite_with_ruy
        # is false, but putting these dependencies in a select() seems to
        # defeat copybara's rewriting rules.
        "@ruy//ruy:context",
        "@ruy//ruy:thread_pool",
        "@gemmlowp",
    ],
)

cc_test(
    name = "cpu_backend_threadpool_test",
    srcs = ["cpu_backend_threadpool_test.cc"],
    deps = [
        ":cpu_backend_context",
        ":cpu_backend_threadpool",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cpu_backend_gemm",
    srcs = [
        "cpu_backend_gemm_custom_gemv.h",
        "cpu_backend_gemm_eigen.cc",
        "cpu_backend_gemm_eigen.h",
        "cpu_backend_gemm_gemmlowp.h",
        "cpu_backend_gemm_x86.h",
    ],
    hdrs = [
        "cpu_backend_gemm.h",
        "cpu_backend_gemm_params.h",
        "cpu_backend_gemm_ruy.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":tflite_with_ruy",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:cpu_check",
        "//tensorflow/lite/kernels/internal:types",
        ":cpu_backend_context",
        ":cpu_backend_threadpool",
        # Depend on ruy regardless of `tflite_with_ruy`. See the comment in
        # cpu_backend_gemm.h about why ruy is the generic path.
        "@ruy//ruy",
        "@ruy//ruy:matrix",
        "@ruy//ruy:path",
        "@ruy//ruy:mul_params",
        "@ruy//ruy/profiler:instrumentation",
        # We only need to depend on gemmlowp and Eigen when tflite_with_ruy
        # is false, but putting these dependencies in a select() seems to
        # defeat copybara's rewriting rules.
        "@gemmlowp",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "cpu_backend_gemm_test",
    srcs = ["cpu_backend_gemm_test.cc"],
    tags = ["notsan"],
    deps = [
        ":cpu_backend_context",
        ":cpu_backend_gemm",
        "@com_google_googletest//:gtest_main",
        "@ruy//ruy:matrix",
        # ruy:reference_mul provides the reference implementation
        # that this test compares against.
        "@ruy//ruy:reference_mul",
    ],
)

cc_library(
    name = "op_macros",
    hdrs = [
        "op_macros.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_library(
    name = "kernel_util",
    srcs = [
        "kernel_util.cc",
    ],
    hdrs = [
        "kernel_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/kernels/internal:quantization_util",
    ],
)

cc_test(
    name = "kernel_util_test",
    size = "small",
    srcs = ["kernel_util_test.cc"],
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    tags = ["tflite_smoke_test"],
    deps = [
        ":kernel_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/testing:util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "test_util_test",
    size = "small",
    srcs = ["test_util_test.cc"],
    deps = [
        ":test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "test_delegate_providers_lib_test",
    size = "small",
    srcs = ["test_delegate_providers_test.cc"],
    # See details in https://github.com/bazelbuild/bazel/issues/11552 to avoid
    # lazy symbol binding failure on macOS.
    linkstatic = select({
        "//tensorflow:macos": True,
        "//conditions:default": False,
    }),
    deps = [
        ":test_delegate_providers_lib",
        "//tensorflow/lite/tools/delegates:default_execution_provider",
        "//tensorflow/lite/tools/delegates:nnapi_delegate_provider",
        "//tensorflow/lite/tools/delegates:xnnpack_delegate_provider",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "padding",
    srcs = [],
    hdrs = ["padding.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:types",
    ],
)

# See also VARIABLE_KERNEL_SRCS below.
BUILTIN_KERNEL_SRCS = [
    "activations.cc",
    "add.cc",
    "add_n.cc",
    "arg_min_max.cc",
    "audio_spectrogram.cc",
    "basic_rnn.cc",
    "batch_matmul.cc",
    "batch_to_space_nd.cc",
    "bidirectional_sequence_lstm.cc",
    "bidirectional_sequence_rnn.cc",
    "broadcast_args.cc",
    "broadcast_to.cc",
    "bucketize.cc",
    "call_once.cc",
    "cast.cc",
    "ceil.cc",
    "comparisons.cc",
    "complex_support.cc",
    "concatenation.cc",
    "conv.cc",
    "conv3d.cc",
    "conv3d_transpose.cc",
    "cumsum.cc",
    "densify.cc",
    "depth_to_space.cc",
    "depthwise_conv.cc",
    "dequantize.cc",
    "detection_postprocess.cc",
    "div.cc",
    "dynamic_update_slice.cc",
    "elementwise.cc",
    "embedding_lookup.cc",
    "embedding_lookup_sparse.cc",
    "exp.cc",
    "expand_dims.cc",
    "fake_quant.cc",
    "fill.cc",
    "floor.cc",
    "floor_div.cc",
    "floor_mod.cc",
    "fully_connected.cc",
    "gather.cc",
    "gather_nd.cc",
    "hashtable.cc",
    "hashtable_find.cc",
    "hashtable_import.cc",
    "hashtable_lookup.cc",
    "hashtable_size.cc",
    "if.cc",
    "l2norm.cc",
    "local_response_norm.cc",
    "logical.cc",
    "lsh_projection.cc",
    "lstm.cc",
    "matrix_diag.cc",
    "matrix_set_diag.cc",
    "maximum_minimum.cc",
    "mfcc.cc",
    "mirror_pad.cc",
    "mul.cc",
    "neg.cc",
    "non_max_suppression.cc",
    "numeric_verify.cc",
    "one_hot.cc",
    "pack.cc",
    "pad.cc",
    "pooling.cc",
    "pow.cc",
    "quantize.cc",
    "random_ops.cc",
    "range.cc",
    "rank.cc",
    "reduce.cc",
    "reshape.cc",
    "resize_bilinear.cc",
    "resize_nearest_neighbor.cc",
    "reverse.cc",
    "reverse_sequence.cc",
    "round.cc",
    "scatter_nd.cc",
    "segment_sum.cc",
    "select.cc",
    "shape.cc",
    "skip_gram.cc",
    "slice.cc",
    "space_to_batch_nd.cc",
    "space_to_depth.cc",
    "sparse_to_dense.cc",
    "split.cc",
    "split_v.cc",
    "squared_difference.cc",
    "squeeze.cc",
    "strided_slice.cc",
    "sub.cc",
    "svdf.cc",
    "tile.cc",
    "topk_v2.cc",
    "transpose.cc",
    "transpose_conv.cc",
    "unidirectional_sequence_lstm.cc",
    "unidirectional_sequence_rnn.cc",
    "unique.cc",
    "unpack.cc",
    "where.cc",
    "while.cc",
    "zeros_like.cc",
    "rfft2d.cc",
]

VARIABLE_KERNEL_SRCS = [
    "assign_variable.cc",
    "read_variable.cc",
    "var_handle.cc",
]

BUILTIN_KERNEL_DEPS = [
    ":cpu_backend_context",
    ":cpu_backend_gemm",
    ":cpu_backend_threadpool",
    ":kernel_util",
    ":lstm_eval",
    ":lstm_shared",
    ":op_macros",
    ":padding",
    "//third_party/eigen3",
    "@flatbuffers",
    "//tensorflow/lite:framework_stable",
    "//tensorflow/lite:minimal_logging",
    "//tensorflow/lite:string_util",
    "//tensorflow/lite:util",
    "//tensorflow/lite/c:common",
    "//tensorflow/lite/kernels/internal:audio_utils",
    "//tensorflow/lite/kernels/internal:common",
    "//tensorflow/lite/kernels/internal:compatibility",
    "//tensorflow/lite/kernels/internal:cpu_check",
    "//tensorflow/lite/kernels/internal:kernel_utils",
    "//tensorflow/lite/kernels/internal:optimized_base",
    "//tensorflow/lite/kernels/internal:quantization_util",
    "//tensorflow/lite/kernels/internal:reference_base",
    "//tensorflow/lite/kernels/internal:strided_slice_logic",
    "//tensorflow/lite/kernels/internal:tensor",
    "//tensorflow/lite/kernels/internal:tensor_utils",
    "//tensorflow/lite/kernels/internal:types",
] + select({
    ":tflite_with_ruy_explicit_true": [],
    # Eigen multi-therading optimizations are only used when ruy is disabled.
    "//conditions:default": [
        ":eigen_support",
        "//tensorflow/lite/kernels/internal:optimized_eigen",
    ],
})

cc_library(
    name = "builtin_op_kernels",
    srcs = BUILTIN_KERNEL_SRCS,
    hdrs = [
        "dequantize.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tf_opts_nortti_if_android() + EXTRA_EIGEN_COPTS,
    visibility = ["//visibility:private"],
    deps = BUILTIN_KERNEL_DEPS + [
        ":variable_op_kernels",
        "@fft2d",
        "@ruy//ruy/profiler:instrumentation",
        "//tensorflow/core/lib/random:philox_random",
        "//tensorflow/core/lib/random:random_distributions_utils",
        "//tensorflow/lite/c:c_api_types",
        # TODO(b/179298174): Move out from the experimental directory.
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite:string",
        "@farmhash_archive//:farmhash",
        "//third_party/fft2d:fft2d_headers",
    ],
)

cc_library(
    name = "variable_op_kernels",
    srcs = VARIABLE_KERNEL_SRCS,
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":kernel_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/kernels/internal:tensor",
        "@flatbuffers",
    ],
)

cc_test(
    name = "variable_ops_test",
    size = "small",
    srcs = [
        "variable_ops_test.cc",
    ],
    deps = [
        ":builtin_ops",
        ":test_main",
        ":variable_op_kernels",  # buildcleaner: keep
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:tensor",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

pybind_extension(
    name = "pywrap_variable_ops",
    srcs = [
        "variable_ops_wrapper.cc",
    ],
    additional_exported_symbols = ["VariableOpsRegisterer"],
    link_in_framework = True,
    module_name = "pywrap_variable_ops",
    deps = [
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:mutable_op_resolver",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cc_library(
    name = "gru_cell",
    srcs = ["gru_cell.cc"],
    hdrs = ["gru_cell.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "custom_ops",
    srcs = [
        "atan2.cc",
        "irfft2d.cc",
        "multinomial.cc",
        "pooling3d.cc",
        "random_standard_normal_custom.cc",
        "random_uniform_custom.cc",
        "roll.cc",
        "sign.cc",
        "table.cc",
        "unidirectional_sequence_gru.cc",
    ],
    hdrs = ["custom_ops_register.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":gru_cell",
        ":kernel_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "//third_party/fft2d:fft2d_headers",
        "@fft2d",
        "@flatbuffers",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "lstm_eval",
    srcs = ["lstm_eval.cc"],
    hdrs = ["lstm_eval.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":cpu_backend_context",
        ":op_macros",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:kernel_utils",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "lstm_shared",
    hdrs = ["lstm_shared.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

# For internal usage by shared libraries only.
cc_library(
    name = "builtin_ops_all_linked",
    srcs = ["register.cc"],
    hdrs = [
        "builtin_op_kernels.h",
        "fully_connected.h",
        "register.h",
    ],
    copts = tflite_copts(),
    # Limit visibility to TFLite only.
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":builtin_op_kernels",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:tflite_with_xnnpack_optional",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

cc_library(
    name = "builtin_ops",
    srcs = ["register.cc"],
    hdrs = [
        "builtin_op_kernels.h",
        "fully_connected.h",
        "register.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":builtin_op_kernels",  # buildcleaner: keep
        "//tensorflow/lite:cc_api_stable",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:tflite_with_xnnpack_optional",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

# The builtin_ops target will resolve to optimized kernels when available. This
# target uses reference kernels only, and is useful for validation and testing.
# It should *not* generally be used in production.
cc_library(
    name = "reference_ops",
    srcs = ["register_ref.cc"],
    hdrs = ["register_ref.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":builtin_op_kernels",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "audio_spectrogram_test",
    size = "small",
    srcs = ["audio_spectrogram_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "mfcc_test",
    size = "small",
    srcs = ["mfcc_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "detection_postprocess_test",
    size = "small",
    srcs = ["detection_postprocess_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "activations_test",
    size = "small",
    srcs = ["activations_test.cc"],
    tags = [
        "tflite_nnapi",
        "tflite_xnnpack",
    ],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "add_test",
    size = "small",
    srcs = ["add_test.cc"],
    tags = [
        "tflite_nnapi",
        "tflite_xnnpack",
    ],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "add_n_test",
    size = "small",
    srcs = ["add_n_test.cc"],
    deps = [
        ":add_n_test_common",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "add_n_test_common",
    testonly = 1,
    hdrs = [
        "add_n_test_common.h",
    ],
    deps = [
        ":test_util",
    ],
)

cc_test(
    name = "arg_min_max_test",
    size = "small",
    srcs = ["arg_min_max_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "div_test",
    size = "small",
    srcs = ["div_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "sub_test",
    size = "small",
    srcs = ["sub_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "transpose_test",
    size = "small",
    srcs = ["transpose_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:reference",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "space_to_batch_nd_test",
    size = "small",
    srcs = ["space_to_batch_nd_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "batch_to_space_nd_test",
    size = "small",
    srcs = ["batch_to_space_nd_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "batch_matmul_test",
    size = "small",
    srcs = ["batch_matmul_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "broadcast_args_test",
    size = "small",
    srcs = ["broadcast_args_test.cc"],
    deps = [
        ":builtin_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "broadcast_to_test",
    size = "small",
    srcs = ["broadcast_to_test.cc"],
    deps = [
        ":builtin_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "bucketize_test",
    size = "small",
    srcs = ["bucketize_test.cc"],
    deps = [
        ":builtin_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "cast_test",
    size = "small",
    srcs = ["cast_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "concatenation_test",
    size = "small",
    srcs = ["concatenation_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "conv_test",
    size = "small",
    srcs = ["conv_test.cc"],
    # TODO(b/162870360): Re-enable nnapi test after delegating grouped conv is added.
    # tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "conv_mem_test",
    srcs = ["conv_mem_test.cc"],
    data = [
        "//tensorflow/lite:testdata/conv3d_huge_im2col.bin",
        "//tensorflow/lite:testdata/conv_huge_im2col.bin",
    ],
    tags = [
        "tflite_not_portable_ios",
        "tflite_smoke_test",
    ],
    deps = [
        ":kernel_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/tools:logging",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "conv3d_test",
    size = "small",
    srcs = ["conv3d_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "densify_test",
    size = "small",
    srcs = ["densify_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "depthwise_conv_hybrid_test",
    size = "small",
    srcs = ["depthwise_conv_hybrid_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "depthwise_conv_test",
    size = "small",
    srcs = ["depthwise_conv_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "dequantize_test",
    size = "small",
    srcs = ["dequantize_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "numeric_verify_test",
    size = "small",
    srcs = ["numeric_verify_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":kernel_util",
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/kernels/internal:reference",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "basic_rnn_test",
    size = "small",
    srcs = ["basic_rnn_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "bidirectional_sequence_lstm_test",
    size = "small",
    srcs = ["bidirectional_sequence_lstm_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "floor_test",
    size = "small",
    srcs = ["floor_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "ceil_test",
    size = "small",
    srcs = ["ceil_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "round_test",
    size = "small",
    srcs = ["round_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "elementwise_test",
    size = "small",
    srcs = ["elementwise_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unidirectional_sequence_lstm_test",
    size = "small",
    srcs = ["unidirectional_sequence_lstm_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "bidirectional_sequence_rnn_test",
    size = "small",
    srcs = ["bidirectional_sequence_rnn_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "unidirectional_sequence_rnn_test",
    size = "small",
    srcs = ["unidirectional_sequence_rnn_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "l2norm_test",
    size = "small",
    srcs = ["l2norm_test.cc"],
    # TODO(b/143912164): Enable NNAPI test when fix nnapi.
    # tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "exp_test",
    size = "small",
    srcs = ["exp_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "fake_quant_test",
    size = "small",
    srcs = ["fake_quant_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "maximum_minimum_test",
    size = "small",
    srcs = ["maximum_minimum_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "reduce_test_common",
    testonly = 1,
    hdrs = [
        "reduce_test_common.h",
    ],
    deps = [
        ":test_util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "reduce_test",
    size = "small",
    srcs = ["reduce_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":reduce_test_common",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "mul_test",
    size = "small",
    srcs = ["mul_test.cc"],
    tags = [
        "tflite_nnapi",
        "tflite_xnnpack",
    ],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "atan2_test",
    size = "small",
    srcs = ["atan2_test.cc"],
    deps = [
        ":custom_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "multinomial_test",
    size = "small",
    srcs = ["multinomial_test.cc"],
    deps = [
        ":custom_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "roll_test",
    size = "small",
    srcs = ["roll_test.cc"],
    deps = [
        ":custom_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "pad_test",
    size = "small",
    srcs = ["pad_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "random_ops_test",
    size = "small",
    srcs = ["random_ops_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "random_standard_normal_custom_test",
    size = "small",
    srcs = ["random_standard_normal_custom_test.cc"],
    deps = [
        ":custom_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "random_uniform_custom_test",
    size = "small",
    srcs = ["random_uniform_custom_test.cc"],
    deps = [
        ":custom_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "sign_test",
    size = "small",
    srcs = ["sign_test.cc"],
    deps = [
        ":custom_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "reshape_test_common",
    testonly = 1,
    hdrs = [
        "reshape_test_common.h",
    ],
    deps = [
        ":test_util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "reshape_test",
    size = "small",
    srcs = ["reshape_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":reshape_test_common",
        ":test_main",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "gather_test",
    size = "small",
    srcs = ["gather_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "gather_nd_test",
    size = "small",
    srcs = ["gather_nd_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "scatter_nd_test",
    size = "small",
    srcs = ["scatter_nd_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "topk_v2_test",
    size = "small",
    srcs = ["topk_v2_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "resize_bilinear_test",
    size = "small",
    srcs = ["resize_bilinear_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "resize_nearest_neighbor_test",
    size = "small",
    srcs = ["resize_nearest_neighbor_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "svdf_test",
    size = "small",
    srcs = ["svdf_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "embedding_lookup_test",
    size = "small",
    srcs = ["embedding_lookup_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "embedding_lookup_sparse_test",
    size = "small",
    srcs = ["embedding_lookup_sparse_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "fully_connected_test",
    size = "small",
    srcs = ["fully_connected_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":builtin_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "local_response_norm_test",
    size = "small",
    srcs = ["local_response_norm_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "pooling_test",
    size = "small",
    srcs = ["pooling_test.cc"],
    tags = [
        "tflite_nnapi",
        "tflite_xnnpack",
    ],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "pooling3d_test",
    size = "small",
    srcs = ["pooling3d_test.cc"],
    deps = [
        ":custom_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "softmax_test",
    size = "small",
    srcs = ["softmax_test.cc"],
    tags = [
        "tflite_nnapi",
        "tflite_xnnpack",
    ],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "log_softmax_test",
    size = "small",
    srcs = ["log_softmax_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "lsh_projection_test",
    size = "small",
    srcs = ["lsh_projection_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "hashtable_lookup_test",
    size = "small",
    srcs = ["hashtable_lookup_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "lstm_test",
    size = "small",
    srcs = ["lstm_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "lstm_eval_test",
    size = "small",
    srcs = ["lstm_eval_test.cc"],
    deps = [
        ":cpu_backend_context",
        ":lstm_eval",
        ":test_main",
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "skip_gram_test",
    size = "small",
    srcs = ["skip_gram_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "space_to_depth_test",
    size = "small",
    srcs = ["space_to_depth_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "depth_to_space_test",
    size = "small",
    srcs = ["depth_to_space_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "split_test",
    size = "small",
    srcs = ["split_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "split_v_test",
    size = "small",
    srcs = ["split_v_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "squeeze_test",
    size = "small",
    srcs = ["squeeze_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "strided_slice_test",
    size = "small",
    srcs = ["strided_slice_test.cc"],
    tags = [
        "tflite_nnapi",
    ],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "table_test",
    size = "small",
    srcs = ["table_test.cc"],
    deps = [
        ":custom_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "tile_test",
    size = "small",
    srcs = ["tile_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "comparisons_test",
    size = "small",
    srcs = [
        "comparisons_test.cc",
    ],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "neg_test",
    size = "small",
    srcs = ["neg_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "select_test",
    size = "small",
    srcs = [
        "select_test.cc",
    ],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "slice_test",
    size = "small",
    srcs = [
        "slice_test.cc",
    ],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "transpose_conv_test",
    size = "small",
    srcs = ["transpose_conv_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "expand_dims_test",
    size = "small",
    srcs = ["expand_dims_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "sparse_to_dense_test",
    size = "small",
    srcs = ["sparse_to_dense_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "shape_test",
    size = "small",
    srcs = ["shape_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "rank_test",
    size = "small",
    srcs = ["rank_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "pow_test",
    size = "small",
    srcs = ["pow_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels/internal:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "pack_test",
    size = "small",
    srcs = ["pack_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "one_hot_test",
    size = "small",
    srcs = ["one_hot_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "logical_test",
    size = "small",
    srcs = ["logical_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "unpack_test",
    size = "small",
    srcs = ["unpack_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "floor_div_test",
    size = "small",
    srcs = ["floor_div_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "where_test",
    size = "small",
    srcs = ["where_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "zeros_like_test",
    size = "small",
    srcs = ["zeros_like_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "floor_mod_test_common",
    testonly = 1,
    hdrs = [
        "floor_mod_test_common.h",
    ],
    deps = [
        ":test_util",
    ],
)

cc_test(
    name = "floor_mod_test",
    size = "small",
    srcs = ["floor_mod_test.cc"],
    deps = [
        ":floor_mod_test_common",
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "range_test",
    size = "small",
    srcs = ["range_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "squared_difference_test",
    size = "small",
    srcs = ["squared_difference_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "call_once_test",
    size = "small",
    srcs = ["call_once_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":kernel_util",
        ":subgraph_test_util",
        ":test_main",
        ":variable_op_kernels",
        "//tensorflow/lite:framework_stable",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "if_test",
    size = "small",
    srcs = ["if_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":kernel_util",
        ":subgraph_test_util",
        ":test_main",
        "//tensorflow/lite:framework_stable",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "while_test",
    size = "small",
    srcs = ["while_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":subgraph_test_util",
        ":test_main",
        "//tensorflow/lite:framework_stable",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "fill_test",
    size = "small",
    srcs = ["fill_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:string",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "unique_test",
    srcs = ["unique_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "reverse_test",
    size = "small",
    srcs = ["reverse_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "rfft2d_test",
    size = "small",
    srcs = ["rfft2d_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "irfft2d_test",
    size = "small",
    srcs = ["irfft2d_test.cc"],
    deps = [
        ":custom_ops",
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "non_max_suppression_test",
    size = "small",
    srcs = ["non_max_suppression_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_test(
    name = "mirror_pad_test",
    srcs = ["mirror_pad_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "subgraph_test_util",
    testonly = 1,
    srcs = ["subgraph_test_util.cc"],
    hdrs = ["subgraph_test_util.h"],
    deps = [
        ":builtin_ops",
        ":kernel_util",
        ":variable_op_kernels",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:interpreter_test_util",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "subgraph_test_util_test",
    size = "small",
    srcs = ["subgraph_test_util_test.cc"],
    deps = [
        ":kernel_util",
        ":subgraph_test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "reverse_sequence_test",
    size = "small",
    srcs = ["reverse_sequence_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "matrix_diag_test",
    size = "small",
    srcs = ["matrix_diag_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quantize_test",
    size = "small",
    srcs = ["quantize_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "matrix_set_diag_test",
    size = "small",
    srcs = ["matrix_set_diag_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "quant_basic_lstm_test",
    size = "small",
    srcs = ["quant_basic_lstm_test.cc"],
    tags = ["tflite_nnapi"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "segment_sum_test",
    srcs = ["segment_sum_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "complex_support_test",
    srcs = ["complex_support_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "cumsum_test",
    size = "small",
    srcs = ["cumsum_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

exports_files(
    [
        "register.h",
        "builtin_op_kernels.h",
        "fully_connected.h",
    ],
    visibility = ["//tensorflow/lite/core/shims:__subpackages__"],
)

cc_test(
    name = "hashtable_ops_test",
    size = "small",
    srcs = [
        "hashtable_ops_test.cc",
    ],
    deps = [
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "unidirectional_sequence_gru_test",
    size = "small",
    srcs = ["unidirectional_sequence_gru_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":custom_ops",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "conv3d_transpose_test",
    size = "small",
    srcs = ["conv3d_transpose_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:string",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "dynamic_update_slice_test",
    size = "small",
    srcs = ["dynamic_update_slice_test.cc"],
    deps = [
        ":test_main",
        ":test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite_combined(combine_conditions = {"deps": [":test_main"]})

# Note this is created mainly for validating external delegates in OSS.
tflite_combine_cc_tests(
    name = "combined_all_kernel_tests",
    deps_conditions = [":test_main"],
)

# This test uses SimpleAllocator and works nicely with ASAN.
tflite_combine_cc_tests(
    name = "combined_all_kernel_with_simple_allocator_tests",
    args = ["--use_simple_allocator=true"],
    deps_conditions = [":test_main"],
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_tests")
