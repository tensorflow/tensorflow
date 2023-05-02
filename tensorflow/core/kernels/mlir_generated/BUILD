# Generates CUDA kernels using MLIR codegen.

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "//tensorflow/core/kernels/mlir_generated:build_defs.bzl",
    "gpu_kernel_library",
    "if_mlir_generated_experimental_kernels_enabled",
    "if_mlir_generated_gpu_kernels_enabled",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
)
load("//tensorflow:tensorflow.default.bzl", "tf_cuda_cc_test", "tf_kernel_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/kernels:__subpackages__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//waymo/...",
    ],
)

bool_flag(
    name = "enable_gpu",
    build_setting_default = True,
)

config_setting(
    name = "is_gpu_enabled",
    flag_values = {":enable_gpu": "True"},
)

# This flag may only be enabled with enable_gpu are true.
bool_flag(
    name = "enable_experimental",
    build_setting_default = False,
)

config_setting(
    name = "is_experimental_enabled",
    flag_values = {":enable_experimental": "True"},
)

cc_library(
    name = "base_op",
    srcs = ["base_op.cc"],
    hdrs = ["base_op.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:allocation_description_proto_cc",
        "//tensorflow/core/framework:op_requires",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:mlir_c_runner_utils",
    ],
)

cc_library(
    name = "base_gpu_op",
    hdrs = ["base_gpu_op.h"],
    deps = [
        ":base_op",
        "//tensorflow/core:framework",
    ],
)

tf_kernel_library(
    name = "gpu_cwise_unary_op",
    srcs = if_mlir_generated_gpu_kernels_enabled([
        "gpu_op_abs.cc",
        "gpu_op_acos.cc",
        "gpu_op_acosh.cc",
        "gpu_op_angle.cc",
        "gpu_op_asin.cc",
        "gpu_op_asinh.cc",
        "gpu_op_atan.cc",
        "gpu_op_atanh.cc",
        "gpu_op_ceil.cc",
        "gpu_op_complex_abs.cc",
        "gpu_op_conj.cc",
        "gpu_op_cos.cc",
        "gpu_op_cosh.cc",
        "gpu_op_digamma.cc",
        "gpu_op_erf.cc",
        "gpu_op_erfc.cc",
        "gpu_op_exp.cc",
        "gpu_op_expm1.cc",
        "gpu_op_floor.cc",
        "gpu_op_imag.cc",
        "gpu_op_invert.cc",
        "gpu_op_is_finite.cc",
        "gpu_op_is_inf.cc",
        "gpu_op_is_nan.cc",
        "gpu_op_lgamma.cc",
        "gpu_op_log.cc",
        "gpu_op_log1p.cc",
        "gpu_op_logical_not.cc",
        "gpu_op_neg.cc",
        "gpu_op_real.cc",
        "gpu_op_reciprocal.cc",
        "gpu_op_rint.cc",
        "gpu_op_round.cc",
        "gpu_op_rsqrt.cc",
        "gpu_op_sigmoid.cc",
        "gpu_op_sign.cc",
        "gpu_op_sin.cc",
        "gpu_op_sinh.cc",
        "gpu_op_sqrt.cc",
        "gpu_op_square.cc",
        "gpu_op_tan.cc",
        "gpu_op_tanh.cc",
    ]),
    copts = if_mlir_generated_experimental_kernels_enabled([
        "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
    ]),
    tags = ["manual"],
    deps = if_mlir_generated_gpu_kernels_enabled([
        ":base_gpu_op",
        ":gpu_angle_kernels",
        ":gpu_ceil_kernels",
        ":gpu_complex_abs_kernels",
        ":gpu_conj_kernels",
        ":gpu_digamma_kernels",
        ":gpu_erf_kernels",
        ":gpu_erfc_kernels",
        ":gpu_exp_kernels",
        ":gpu_expm1_kernels",
        ":gpu_floor_kernels",
        ":gpu_imag_kernels",
        ":gpu_invert_kernels",
        ":gpu_is_finite_kernels",
        ":gpu_is_inf_kernels",
        ":gpu_is_nan_kernels",
        ":gpu_lgamma_kernels",
        ":gpu_log_kernels",
        ":gpu_log1p_kernels",
        ":gpu_logical_not_kernels",
        ":gpu_neg_kernels",
        ":gpu_real_kernels",
        ":gpu_reciprocal_kernels",
        ":gpu_rint_kernels",
        ":gpu_round_kernels",
        ":gpu_rsqrt_kernels",
        ":gpu_sigmoid_kernels",
        ":gpu_sign_kernels",
        ":gpu_sqrt_kernels",
        ":gpu_square_kernels",
        "//third_party/eigen3",
    ]) + if_mlir_generated_experimental_kernels_enabled(
        [
            ":gpu_acos_kernels_experimental",
            ":gpu_acosh_kernels_experimental",
            ":gpu_asin_kernels_experimental",
            ":gpu_asinh_kernels_experimental",
            ":gpu_atan_kernels_experimental",
            ":gpu_cos_kernels_experimental",
            ":gpu_cosh_kernels_experimental",
            ":gpu_sin_kernels_experimental",
            ":gpu_sinh_kernels_experimental",
            ":gpu_tan_kernels_experimental",
            ":gpu_tanh_kernels_experimental",
            ":gpu_abs_kernels_experimental",
            ":gpu_atanh_kernels_experimental",
        ],
        [
            ":gpu_abs_kernels",
            ":gpu_atanh_kernels",
            ":gpu_acos_kernels",
            ":gpu_acosh_kernels",
            ":gpu_asin_kernels",
            ":gpu_asinh_kernels",
            ":gpu_atan_kernels",
            ":gpu_cos_kernels",
            ":gpu_cosh_kernels",
            ":gpu_sin_kernels",
            ":gpu_sinh_kernels",
            ":gpu_tan_kernels",
            ":gpu_tanh_kernels",
        ],
    ),
)

tf_kernel_library(
    name = "gpu_cwise_binary_op",
    srcs = if_mlir_generated_gpu_kernels_enabled([
        "gpu_op_add.cc",
        "gpu_op_atan2.cc",
        "gpu_op_bitwise_and.cc",
        "gpu_op_bitwise_or.cc",
        "gpu_op_bitwise_xor.cc",
        "gpu_op_complex.cc",
        "gpu_op_truncate_div.cc",
        "gpu_op_div.cc",
        "gpu_op_div_no_nan.cc",
        "gpu_op_equal.cc",
        "gpu_op_floor_div.cc",
        "gpu_op_floor_mod.cc",
        "gpu_op_greater.cc",
        "gpu_op_greater_equal.cc",
        "gpu_op_left_shift.cc",
        "gpu_op_less.cc",
        "gpu_op_less_equal.cc",
        "gpu_op_logical_and.cc",
        "gpu_op_logical_or.cc",
        "gpu_op_maximum.cc",
        "gpu_op_minimum.cc",
        "gpu_op_mul.cc",
        "gpu_op_mul_no_nan.cc",
        "gpu_op_not_equal.cc",
        "gpu_op_polygamma.cc",
        "gpu_op_pow.cc",
        "gpu_op_right_shift.cc",
        "gpu_op_select.cc",
        "gpu_op_squared_difference.cc",
        "gpu_op_sub.cc",
        "gpu_op_xdivy.cc",
        "gpu_op_xlog1py.cc",
        "gpu_op_xlogy.cc",
        "gpu_op_zeta.cc",
    ]),
    copts = if_mlir_generated_experimental_kernels_enabled([
        "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
    ]),
    tags = ["manual"],
    deps = if_mlir_generated_gpu_kernels_enabled([
        ":base_gpu_op",
        ":gpu_atan2_kernels",
        ":gpu_bitwise_and_kernels",
        ":gpu_bitwise_or_kernels",
        ":gpu_bitwise_xor_kernels",
        ":gpu_complex_kernels",
        ":gpu_truncate_div_kernels",
        ":gpu_floor_mod_kernels",
        ":gpu_left_shift_kernels",
        ":gpu_logical_and_kernels",
        ":gpu_logical_or_kernels",
        ":gpu_maximum_kernels",
        ":gpu_minimum_kernels",
        ":gpu_polygamma_kernels",
        ":gpu_pow_kernels",
        ":gpu_right_shift_kernels",
        ":gpu_select_v2_kernels",
        ":gpu_squared_difference_kernels",
        ":gpu_xdivy_kernels",
        ":gpu_xlog1py_kernels",
        ":gpu_xlogy_kernels",
        ":gpu_zeta_kernels",
        "//third_party/eigen3",
    ]) + if_mlir_generated_experimental_kernels_enabled(
        [
            ":gpu_add_v2_kernels_experimental",
            ":gpu_sub_kernels_experimental",
            ":gpu_div_kernels_experimental",
            ":gpu_div_no_nan_kernels_experimental",
            ":gpu_mul_kernels_experimental",
            ":gpu_mul_no_nan_kernels_experimental",
            ":gpu_floor_div_kernels_experimental",
            ":gpu_equal_kernels_experimental",
            ":gpu_not_equal_kernels_experimental",
            ":gpu_greater_kernels_experimental",
            ":gpu_greater_equal_kernels_experimental",
            ":gpu_less_equal_kernels_experimental",
            ":gpu_less_kernels_experimental",
        ],
        [
            ":gpu_add_v2_kernels",
            ":gpu_sub_kernels",
            ":gpu_div_kernels",
            ":gpu_div_no_nan_kernels",
            ":gpu_mul_kernels",
            ":gpu_mul_no_nan_kernels",
            ":gpu_floor_div_kernels",
            ":gpu_equal_kernels",
            ":gpu_not_equal_kernels",
            ":gpu_greater_kernels",
            ":gpu_greater_equal_kernels",
            ":gpu_less_equal_kernels",
            ":gpu_less_kernels",
        ],
    ),
)

tf_kernel_library(
    name = "cwise_op",
    srcs = [],
    visibility = [
        ":friends",
        "//tensorflow/core/kernels:__subpackages__",
    ],
    deps = if_cuda_or_rocm([
        ":gpu_cwise_unary_op",
        ":gpu_cwise_binary_op",
    ]),
)

tf_kernel_library(
    name = "gpu_cast_op",
    srcs = if_mlir_generated_gpu_kernels_enabled([
        "gpu_op_cast.cc",
    ]),
    tags = ["manual"],
    deps = if_mlir_generated_gpu_kernels_enabled([
        ":base_gpu_op",
        ":gpu_cast_kernels",
        "//third_party/eigen3",
    ]),
)

tf_kernel_library(
    name = "cast_op",
    srcs = [],
    deps = [
    ] + if_cuda_or_rocm([
        ":gpu_cast_op",
    ]),
)

tf_kernel_library(
    name = "gpu_constant_op",
    srcs = if_mlir_generated_gpu_kernels_enabled([
        "gpu_op_ones_like.cc",
        "gpu_op_zeros_like.cc",
    ]),
    copts = if_mlir_generated_experimental_kernels_enabled([
        "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
    ]),
    tags = ["manual"],
    deps = if_mlir_generated_gpu_kernels_enabled([
        ":base_gpu_op",
        ":gpu_ones_like_kernels",
        ":gpu_zeros_like_kernels",
        "//third_party/eigen3",
    ]),
)

tf_kernel_library(
    name = "constant_op",
    srcs = [],
    deps = [
    ] + if_cuda_or_rocm([
        ":gpu_constant_op",
    ]),
)

tf_kernel_library(
    name = "gpu_nextafter_op",
    srcs = if_mlir_generated_gpu_kernels_enabled([
        "gpu_op_next_after.cc",
    ]),
    tags = ["manual"],
    deps = if_mlir_generated_gpu_kernels_enabled([
        ":base_gpu_op",
        ":gpu_next_after_kernels",
    ]),
)

tf_kernel_library(
    name = "nextafter_op",
    srcs = [],
    deps = [] + if_cuda_or_rocm([":gpu_nextafter_op"]),
)

tf_kernel_library(
    name = "gpu_relu_op",
    srcs = if_mlir_generated_gpu_kernels_enabled([
        "gpu_op_elu.cc",
        "gpu_op_relu.cc",
        "gpu_op_selu.cc",
    ]),
    copts = if_mlir_generated_experimental_kernels_enabled([
        "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
    ]),
    tags = ["manual"],
    deps = if_mlir_generated_gpu_kernels_enabled([
        ":base_gpu_op",
        ":gpu_elu_kernels",
        ":gpu_relu_kernels",
        ":gpu_selu_kernels",
        "//third_party/eigen3",
    ]),
)

tf_kernel_library(
    name = "relu_op",
    srcs = [],
    deps = [
    ] + if_cuda_or_rocm([
        ":gpu_relu_op",
    ]),
)

tf_kernel_library(
    name = "gpu_softplus_op",
    srcs = if_mlir_generated_gpu_kernels_enabled([
        "gpu_op_softplus.cc",
    ]),
    tags = ["manual"],
    deps = if_mlir_generated_gpu_kernels_enabled([
        ":base_gpu_op",
        ":gpu_softplus_kernels",
        "//third_party/eigen3",
    ]),
)

tf_kernel_library(
    name = "softplus_op",
    srcs = [],
    deps = [
    ] + if_cuda_or_rocm([
        ":gpu_softplus_op",
    ]),
)

tf_kernel_library(
    name = "gpu_softsign_op",
    srcs = if_mlir_generated_gpu_kernels_enabled([
        "gpu_op_softsign.cc",
    ]),
    tags = ["manual"],
    deps = if_mlir_generated_gpu_kernels_enabled([
        ":base_gpu_op",
        ":gpu_softsign_kernels",
        "//third_party/eigen3",
    ]),
)

tf_kernel_library(
    name = "softsign_op",
    srcs = [],
    deps = [
    ] + if_cuda_or_rocm([
        ":gpu_softsign_op",
    ]),
)

cc_library(
    name = "base_ops_test",
    testonly = 1,
    srcs = ["base_ops_test.cc"],
    hdrs = ["base_ops_test.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:tensorflow",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "base_unary_ops_test",
    testonly = 1,
    hdrs = ["base_unary_ops_test.h"],
    deps = [
        ":base_ops_test",
        "//tensorflow/compiler/mlir/tools/kernel_gen:tf_jit_cache",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
    ],
)

tf_cuda_cc_test(
    name = "gpu_unary_ops_test",
    size = "medium",
    srcs = ["gpu_unary_ops_test.cc"],
    extra_copts = if_mlir_generated_experimental_kernels_enabled([
        "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
    ]) + if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    shard_count = 20,
    tags = tf_cuda_tests_tags() + [
        "no_cuda",  # TODO(b/196608406): re-enable
        "no_cuda_asan",  # TODO(b/171341759): re-enable.
        "no_rocm", # fail on CI
    ],
    deps = [
        ":base_ops_test",
        ":base_unary_ops_test",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
    ],
)

tf_cuda_cc_test(
    name = "gpu_unary_ops_large_tensor_test",
    size = "large",
    srcs = ["gpu_unary_ops_large_tensor_test.cc"],
    extra_copts = if_mlir_generated_experimental_kernels_enabled([
        "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
    ]) + if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    tags = tf_cuda_tests_tags() + [
        "no_cuda",  # TODO(b/196608406): re-enable
        "no_cuda_asan",  # TODO(b/171341759): re-enable.
    ],
    deps = [
        ":base_ops_test",
        ":base_unary_ops_test",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
    ],
)

cc_library(
    name = "base_binary_ops_test",
    testonly = 1,
    hdrs = ["base_binary_ops_test.h"],
    deps = [
        ":base_ops_test",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
    ],
)

tf_cuda_cc_test(
    name = "gpu_binary_ops_test",
    size = "medium",
    srcs = ["gpu_binary_ops_test.cc"],
    extra_copts = if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ) + if_mlir_generated_experimental_kernels_enabled(
        [
            "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
        ],
    ),
    shard_count = 20,
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # b/173033461
        "no_rocm",  # failed since 7de9cf4
    ],
    deps = [
        ":base_binary_ops_test",
        ":base_ops_test",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
    ],
)

tf_cuda_cc_test(
    name = "gpu_binary_ops_large_tensor_test",
    size = "large",
    srcs = ["gpu_binary_ops_large_tensor_test.cc"],
    extra_copts = if_mlir_generated_experimental_kernels_enabled([
        "-DMLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED",
    ]) + if_mlir_generated_gpu_kernels_enabled(
        ["-DMLIR_GENERATED_GPU_KERNELS_ENABLED"],
    ),
    tags = tf_cuda_tests_tags() + [
        "no_cuda",  # TODO(b/196608406): re-enable
        "no_cuda_asan",  # TODO(b/171341759): re-enable.
    ],
    deps = [
        ":base_binary_ops_test",
        ":base_ops_test",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
    ],
)

# Trigonometric kernels.
gpu_kernel_library(
    name = "gpu_acos_kernels",
    op = "acos",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    # Cannot vectorize.
)

gpu_kernel_library(
    name = "gpu_acos_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    op = "acos",
    tile_size = "256",
    types = [],
    # Cannot vectorize.
)

gpu_kernel_library(
    name = "gpu_acosh_kernels",
    op = "acosh",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    # May be compute-bound.
)

gpu_kernel_library(
    name = "gpu_acosh_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    op = "acosh",
    tile_size = "256",
    types = [],
    # May be compute-bound.
)

gpu_kernel_library(
    name = "gpu_asin_kernels",
    op = "asin",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    # Cannot vectorize.
)

gpu_kernel_library(
    name = "gpu_asin_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    op = "asin",
    tile_size = "256",
    types = [],
    # Cannot vectorize.
)

gpu_kernel_library(
    name = "gpu_asinh_kernels",
    op = "asinh",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    # Cannot vectorize.
)

gpu_kernel_library(
    name = "gpu_asinh_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    op = "asinh",
    tile_size = "256",
    types = [],
    # Cannot vectorize.
)

gpu_kernel_library(
    name = "gpu_atan_kernels",
    op = "atan",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_atan_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    op = "atan",
    tile_size = "256",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_atanh_kernels",
    op = "atanh",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_atanh_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    op = "atanh",
    tile_size = "256",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_atan2_kernels",
    op = "atan2",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_cos_kernels",
    jit_types = [
        "c64",
        "c128",
    ],
    op = "cos",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    name = "gpu_cos_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    jit_types = [
        "c64",
        "c128",
    ],
    op = "cos",
    tile_size = "256",
    types = [],
)

gpu_kernel_library(
    name = "gpu_cosh_kernels",
    jit_types = [
        "c64",
        "c128",
    ],
    op = "cosh",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    # May be compute-bound.
)

gpu_kernel_library(
    name = "gpu_cosh_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    jit_types = [
        "c64",
        "c128",
    ],
    op = "cosh",
    tile_size = "256",
    types = [],
    # May be compute-bound.
)

gpu_kernel_library(
    name = "gpu_sin_kernels",
    jit_types = [
        "c64",
        "c128",
    ],
    op = "sin",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    name = "gpu_sin_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    jit_types = [
        "c64",
        "c128",
    ],
    op = "sin",
    tile_size = "256",
    types = [],
)

gpu_kernel_library(
    name = "gpu_sinh_kernels",
    jit_types = [
        "c64",
        "c128",
    ],
    op = "sinh",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    # May be compute-bound.
)

gpu_kernel_library(
    name = "gpu_sinh_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    jit_types = [
        "c64",
        "c128",
    ],
    op = "sinh",
    tile_size = "256",
    types = [],
    # May be compute-bound.
)

gpu_kernel_library(
    name = "gpu_tan_kernels",
    jit_types = [
        "c64",
        "c128",
    ],
    op = "tan",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    name = "gpu_tan_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    jit_types = [
        "c64",
        "c128",
    ],
    op = "tan",
    tile_size = "256",
    types = [],
)

gpu_kernel_library(
    name = "gpu_tanh_kernels",
    op = "tanh",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_tanh_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
    ],
    op = "tanh",
    tile_size = "256",
    types = [],
    unroll_factors = "4",
)

# Rounding kernels

gpu_kernel_library(
    name = "gpu_ceil_kernels",
    op = "ceil",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_floor_kernels",
    op = "floor",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_floor_mod_kernels",
    jit_types = [
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
        "f16",
        "f32",
        "f64",
    ],
    op = "floor_mod",
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_rint_kernels",
    jit_types = ["f16"],
    op = "rint",
    tile_size = "1024",
    types = [
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    name = "gpu_round_kernels",
    op = "round",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i32",
        "i64",
    ],
)

# Predicate kernels

gpu_kernel_library(
    name = "gpu_equal_kernels",
    op = "equal",
    output_types = ["i1"] * 10,
    tile_size = "1024",
    types = [
        "c64",
        "c128",
        "f16",
        "f32",
        "f64",
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_equal_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "c64",
        "c128",
        "f16",
        "f32",
        "f64",
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    op = "equal",
    output_jit_i64_indexed_for_large_tensors_types = ["i1"] * 10,
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_greater_kernels",
    op = "greater",
    output_types = ["i1"] * 10,
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_greater_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "greater",
    output_jit_i64_indexed_for_large_tensors_types = ["i1"] * 10,
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_greater_equal_kernels",
    op = "greater_equal",
    output_types = ["i1"] * 10,
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_greater_equal_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "greater_equal",
    output_jit_i64_indexed_for_large_tensors_types = ["i1"] * 10,
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_is_finite_kernels",
    op = "is_finite",
    output_types = ["i1"] * 3,
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_is_inf_kernels",
    op = "is_inf",
    output_types = ["i1"] * 3,
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_is_nan_kernels",
    op = "is_nan",
    output_types = ["i1"] * 3,
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_less_kernels",
    op = "less",
    output_types = ["i1"] * 10,
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_less_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "less",
    output_jit_i64_indexed_for_large_tensors_types = ["i1"] * 10,
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_less_equal_kernels",
    op = "less_equal",
    output_types = ["i1"] * 10,
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_less_equal_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "less_equal",
    output_jit_i64_indexed_for_large_tensors_types = ["i1"] * 10,
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_not_equal_kernels",
    op = "not_equal",
    output_types = ["i1"] * 10,
    tile_size = "1024",
    types = [
        "c64",
        "c128",
        "f16",
        "f32",
        "f64",
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_not_equal_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "c64",
        "c128",
        "f16",
        "f32",
        "f64",
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    op = "not_equal",
    output_jit_i64_indexed_for_large_tensors_types = ["i1"] * 10,
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

# Complex-specifc kernels

gpu_kernel_library(
    name = "gpu_angle_kernels",
    op = "angle",
    output_types = [
        "f32",
        "f64",
    ],
    tile_size = "256",
    types = [
        "c64",
        "c128",
    ],
    unroll_factors = "2",
)

gpu_kernel_library(
    name = "gpu_complex_abs_kernels",
    op = "complex_abs",
    output_types = [
        "f32",
        "f64",
    ],
    tile_size = "256",
    types = [
        "c64",
        "c128",
    ],
)

gpu_kernel_library(
    name = "gpu_complex_kernels",
    op = "complex",
    output_types = [
        "c64",
        "c128",
    ],
    tile_size = "1024",
    types = [
        "f32",
        "f64",
    ],
    unroll_factors = "2",
)

gpu_kernel_library(
    name = "gpu_conj_kernels",
    op = "conj",
    tile_size = "256",
    types = [
        "c64",
        "c128",
    ],
    unroll_factors = "2",
)

gpu_kernel_library(
    name = "gpu_imag_kernels",
    op = "imag",
    output_types = [
        "f32",
        "f64",
    ],
    tile_size = "256",
    types = [
        "c64",
        "c128",
    ],
)

gpu_kernel_library(
    name = "gpu_real_kernels",
    op = "real",
    output_types = [
        "f32",
        "f64",
    ],
    tile_size = "256",
    types = [
        "c64",
        "c128",
    ],
)

# Arithmetic kernels

# TODO(b/25387198): Add an int32 kernel.
gpu_kernel_library(
    name = "gpu_abs_kernels",
    jit_types = [
        "i8",
        "i16",
    ],
    op = "abs",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_abs_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "i64",
    ],
    jit_types = [
        "i8",
        "i16",
    ],
    op = "abs",
    tile_size = "256",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_add_v2_kernels",
    op = "add_v2",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "c64",
        "c128",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_add_v2_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "c64",
        "c128",
    ],
    op = "add_v2",
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_div_kernels",
    jit_types = [
        "i8",
        "ui32",
        "ui64",
    ],
    op = "div",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "c64",
        "c128",
    ],
    unroll_factors = "16B",
)

gpu_kernel_library(
    name = "gpu_div_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "c64",
        "c128",
    ],
    jit_types = [
        "i8",
        "ui32",
        "ui64",
    ],
    op = "div",
    tile_size = "1024",
    types = [],
    unroll_factors = "16B",
)

gpu_kernel_library(
    name = "gpu_div_no_nan_kernels",
    op = "div_no_nan",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
)

gpu_kernel_library(
    name = "gpu_div_no_nan_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
    op = "div_no_nan",
    tile_size = "1024",
    types = [],
)

gpu_kernel_library(
    name = "gpu_floor_div_kernels",
    jit_types = [
        "i8",
        "ui32",
        "ui64",
    ],
    op = "floor_div",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "i16",
        "i64",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_floor_div_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "i16",
        "i64",
        "f64",
    ],
    jit_types = [
        "i8",
        "ui32",
        "ui64",
    ],
    op = "floor_div",
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_maximum_kernels",
    jit_types = [
        "i8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "maximum",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i16",
        "i64",
        "ui8",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_minimum_kernels",
    jit_types = [
        "i8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "minimum",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i16",
        "i64",
        "ui8",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_mul_kernels",
    op = "mul",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
    # For complex MulOp kernels, we don't use unrolling, it would only cause
    # slowdowns.
    unroll_factors_override = {
        "c64": None,
        "c128": None,
    },
)

gpu_kernel_library(
    name = "gpu_mul_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    op = "mul",
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
    # For complex MulOp kernels, we don't use unrolling, it would only cause
    # slowdowns.
    unroll_factors_override = {
        "c64": None,
        "c128": None,
    },
)

gpu_kernel_library(
    name = "gpu_mul_no_nan_kernels",
    op = "mul_no_nan",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
    unroll_factors = "4",
    # For complex MulNoNanOp kernels, we don't use unrolling, it would only
    # cause slowdowns.
    unroll_factors_override = {
        "c64": None,
        "c128": None,
    },
)

gpu_kernel_library(
    name = "gpu_mul_no_nan_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
    op = "mul_no_nan",
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
    # For complex MulNoNanOp kernels, we don't use unrolling, it would only
    # cause slowdowns.
    unroll_factors_override = {
        "c64": None,
        "c128": None,
    },
)

gpu_kernel_library(
    name = "gpu_neg_kernels",
    op = "neg",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "c64",
        "c128",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_pow_kernels",
    jit_types = [
        "i8",
        "i16",
    ],
    op = "pow",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i64",
    ],
)

gpu_kernel_library(
    name = "gpu_reciprocal_kernels",
    op = "reciprocal",
    tile_size = "256",
    types = [
        "c64",
        "c128",
        "f16",
        "f32",
        "f64",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_sign_kernels",
    jit_types = [
        "i8",
        "i16",
    ],
    op = "sign",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
        "i32",
        "i64",
        "c64",
        "c128",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_sub_kernels",
    jit_types = [
        "i8",
        "i16",
        "ui8",
        "ui16",
    ],
    op = "sub",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i32",
        "i64",
        "c64",
        "c128",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_sub_kernels_experimental",
    jit_i64_indexed_for_large_tensors_types = [
        "f16",
        "f32",
        "f64",
        "i32",
        "i64",
        "c64",
        "c128",
    ],
    jit_types = [
        "i8",
        "i16",
        "ui8",
        "ui16",
    ],
    op = "sub",
    tile_size = "1024",
    types = [],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_truncate_div_kernels",
    jit_types = [
        "i8",
        "ui32",
        "ui64",
        "f16",
        "f32",
        "f64",
    ],
    op = "truncate_div",
    tile_size = "1024",
    types = [],
)

gpu_kernel_library(
    name = "gpu_xdivy_kernels",
    op = "xdivy",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
    unroll_factors = "4",
)

# Logarithmic and exponential kernels
gpu_kernel_library(
    name = "gpu_exp_kernels",
    op = "exp",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_expm1_kernels",
    op = "expm1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_log_kernels",
    op = "log",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_log1p_kernels",
    op = "log1p",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_xlogy_kernels",
    op = "xlogy",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
    unroll_factors = "4",
    # For complex XlogyOp kernels, we don't use unrolling, it would only cause
    # slowdowns.
    unroll_factors_override = {
        "c64": None,
        "c128": None,
    },
)

gpu_kernel_library(
    name = "gpu_xlog1py_kernels",
    op = "xlog1py",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
    unroll_factors = "4",
    # For complex Xlog1pyOp kernels, we don't use unrolling, it would only cause
    # slowdowns.
    unroll_factors_override = {
        "c64": None,
        "c128": None,
    },
)

# Square and square root kernels

gpu_kernel_library(
    name = "gpu_sqrt_kernels",
    op = "sqrt",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_rsqrt_kernels",
    op = "rsqrt",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_square_kernels",
    jit_types = [
        "i8",
        "i16",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "square",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_squared_difference_kernels",
    op = "squared_difference",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i64",
    ],
    unroll_factors = "4",
)

# Bitwise operations.

gpu_kernel_library(
    name = "gpu_bitwise_and_kernels",
    op = "bitwise_and",
    tile_size = "1024",
    types = [
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_bitwise_or_kernels",
    op = "bitwise_or",
    tile_size = "1024",
    types = [
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_bitwise_xor_kernels",
    op = "bitwise_xor",
    tile_size = "1024",
    types = [
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_invert_kernels",
    op = "invert",
    tile_size = "1024",
    types = [
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_left_shift_kernels",
    op = "left_shift",
    tile_size = "1024",
    types = [
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_right_shift_kernels",
    op = "right_shift",
    tile_size = "1024",
    types = [
        "i8",
        "i16",
        "i32",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    unroll_factors = "4",
)

# Logical kernels

gpu_kernel_library(
    name = "gpu_logical_not_kernels",
    op = "logical_not",
    tile_size = "256",
    types = ["i1"],
)

gpu_kernel_library(
    name = "gpu_logical_and_kernels",
    op = "logical_and",
    tile_size = "1024",
    types = [
        "i1",
    ],
)

gpu_kernel_library(
    name = "gpu_logical_or_kernels",
    op = "logical_or",
    tile_size = "1024",
    types = [
        "i1",
    ],
)

# Erf kernels

gpu_kernel_library(
    name = "gpu_erf_kernels",
    op = "erf",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_erfc_kernels",
    op = "erfc",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

# Gamma kernels

gpu_kernel_library(
    name = "gpu_polygamma_kernels",
    op = "polygamma",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    name = "gpu_digamma_kernels",
    op = "digamma",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    name = "gpu_lgamma_kernels",
    op = "lgamma",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    # The zeta kernels needs many registers so tile at 256.
    name = "gpu_zeta_kernels",
    op = "zeta",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    # TODO(b/178388085): Enable unrolling after vectorization is fixed.
    # unroll_factors = "4",
)

# TODO(b/160731748): Re-enable when it works again.
# gpu_kernel_library(
#     name = "gpu_bias_add_kernels",
#     op = "bias_add",
#     tile_size = "16x16",
#     types = [
#         "f16",
#         "f32",
#         "f64",
#     ],
# )

gpu_kernel_library(
    name = "gpu_relu_kernels",
    jit_types = [
        "i8",
        "i16",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "relu",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "16B",
)

gpu_kernel_library(
    name = "gpu_elu_kernels",
    op = "elu",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    name = "gpu_selu_kernels",
    op = "selu",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    name = "gpu_sigmoid_kernels",
    op = "sigmoid",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

# Kernels that support all floating-point types.
[
    gpu_kernel_library(
        name = "gpu_" + op + "_kernels",
        op = op,
        tile_size = "256",
        types = [
            "f16",
            "f32",
            "f64",
        ],
        unroll_factors = "4",
    )
    for op in [
        "relu_grad",
        "softplus",
        "softsign",
    ]
]

gpu_kernel_library(
    name = "gpu_cast_kernels",
    op = "cast",
    # We generate all combinations of input types/output types from the set
    # {i1, i8, i16, i32, i64, ui8, ui16, ui32, ui64, f16, f32, f64, c64, c128}
    # to the set
    # {i1, i8, i16, i32, i64, ui8, ui16, ui32, ui64, f16, f32, f64, c64, c128}.
    # The easiest way to do this is to repeat each input type 14 times and match
    # it with the 14 different output types (thus, the list of 14 different
    # output types needs to be repeated 14 times).
    output_types = [
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ] * 14,
    tile_size = "256",
    types = ["i1"] * 14 + ["i8"] * 14 + ["i16"] * 14 + ["i32"] * 14 +
            ["i64"] * 14 + ["ui8"] * 14 + ["ui16"] * 14 + ["ui32"] * 14 +
            ["ui64"] * 14 + ["f16"] * 14 + ["f32"] * 14 + ["f64"] * 14 +
            ["c64"] * 14 + ["c128"] * 14,
)

gpu_kernel_library(
    name = "gpu_select_v2_kernels",
    jit_types = [
        "i8",
        "i16",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    max_supported_rank = 8,
    op = "select_v2",
    tile_size = "256",
    types = [
        "i1",
        "i32",
        "i64",
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
)

gpu_kernel_library(
    name = "gpu_zeros_like_kernels",
    jit_types = [
        "i8",
        "i16",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "zeros_like",
    tile_size = "1024",
    types = [
        "i1",
        "i64",
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
)

gpu_kernel_library(
    name = "gpu_ones_like_kernels",
    jit_types = [
        "i8",
        "i16",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
    ],
    op = "ones_like",
    tile_size = "1024",
    types = [
        "i1",
        "i64",
        "f16",
        "f32",
        "f64",
        "c64",
        "c128",
    ],
)

gpu_kernel_library(
    name = "gpu_next_after_kernels",
    op = "next_after",
    tile_size = "1024",
    types = [
        "f32",
        "f64",
    ],
)
