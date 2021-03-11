# Generates CUDA kernels using MLIR codegen.

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "//tensorflow/core/kernels/mlir_generated:build_defs.bzl",
    "cpu_kernel_library",
    "gpu_kernel_library",
    "if_mlir_experimental_kernels_enabled",
    "if_mlir_generated_gpu_kernels_enabled",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")  # buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")  # buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")  # buildifier: disable=same-origin-load
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    default_visibility = [
        "//tensorflow/core/kernels:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

bool_flag(
    name = "enable_gpu",
    build_setting_default = True,
)

config_setting(
    name = "is_gpu_enabled",
    flag_values = {":enable_gpu": "True"},
)

bool_flag(
    name = "enable_experimental",
    build_setting_default = False,
)

config_setting(
    name = "is_experimental_enabled",
    flag_values = {":enable_experimental": "True"},
)

filegroup(
    name = "enabled_unary_gpu_kernel_srcs",
    srcs = [
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
        "gpu_op_rsqrt.cc",
        "gpu_op_sign.cc",
        "gpu_op_sin.cc",
        "gpu_op_sinh.cc",
        "gpu_op_sqrt.cc",
        "gpu_op_square.cc",
        "gpu_op_tan.cc",
        "gpu_op_tanh.cc",
    ],
    compatible_with = get_compatible_with_cloud(),
)

filegroup(
    name = "experimental_unary_gpu_kernel_srcs",
    srcs = [
    ],
    compatible_with = get_compatible_with_cloud(),
)

filegroup(
    name = "unary_gpu_kernel_srcs",
    srcs = [
        ":enabled_unary_gpu_kernel_srcs",
    ] + if_mlir_experimental_kernels_enabled(
        if_true = [":experimental_unary_gpu_kernel_srcs"],
    ),
    compatible_with = get_compatible_with_cloud(),
)

filegroup(
    name = "enabled_binary_gpu_kernel_srcs",
    srcs = [
        "gpu_op_add.cc",
        "gpu_op_atan2.cc",
        "gpu_op_bitwise_and.cc",
        "gpu_op_bitwise_or.cc",
        "gpu_op_bitwise_xor.cc",
        "gpu_op_complex.cc",
        "gpu_op_div.cc",
        "gpu_op_equal.cc",
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
        "gpu_op_not_equal.cc",
        "gpu_op_polygamma.cc",
        "gpu_op_pow.cc",
        "gpu_op_right_shift.cc",
        "gpu_op_sub.cc",
        "gpu_op_zeta.cc",
    ],
    compatible_with = get_compatible_with_cloud(),
)

filegroup(
    name = "experimental_binary_gpu_kernel_srcs",
    srcs = [
        "gpu_op_floor_div.cc",
        "gpu_op_squared_difference.cc",
    ],
    compatible_with = get_compatible_with_cloud(),
)

filegroup(
    name = "binary_gpu_kernel_srcs",
    srcs = [
        ":enabled_binary_gpu_kernel_srcs",
    ] + if_mlir_experimental_kernels_enabled(
        if_true = [":experimental_binary_gpu_kernel_srcs"],
    ),
    compatible_with = get_compatible_with_cloud(),
)

cc_library(
    name = "base_op",
    srcs = ["base_op.cc"],
    hdrs = ["base_op.h"],
    compatible_with = get_compatible_with_cloud(),
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
    compatible_with = get_compatible_with_cloud(),
    deps = [":base_op"],
)

cc_library(
    name = "base_cpu_op",
    hdrs = ["base_cpu_op.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [":base_op"],
)

tf_kernel_library(
    name = "gpu_cwise_unary_op",
    srcs = [":unary_gpu_kernel_srcs"],
    tags = ["manual"],
    # Technically we only need to depend on the kernel libraries for the
    # kernels which are enabled by default. But this would make our BUILD
    # target structure uglier. We already need to make sure that those
    # targets can be built, so it should not hurt to link them in even if
    # they are currently not needed yet.
    deps = [
        ":base_gpu_op",
        ":gpu_abs_kernels",
        ":gpu_acos_kernels",
        ":gpu_acosh_kernels",
        ":gpu_angle_kernels",
        ":gpu_asin_kernels",
        ":gpu_asinh_kernels",
        ":gpu_atan_kernels",
        ":gpu_atanh_kernels",
        ":gpu_ceil_kernels",
        ":gpu_complex_abs_kernels",
        ":gpu_conj_kernels",
        ":gpu_cos_kernels",
        ":gpu_cosh_kernels",
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
        ":gpu_log1p_kernels",
        ":gpu_log_kernels",
        ":gpu_logical_not_kernels",
        ":gpu_neg_kernels",
        ":gpu_real_kernels",
        ":gpu_rsqrt_kernels",
        ":gpu_sign_kernels",
        ":gpu_sin_kernels",
        ":gpu_sinh_kernels",
        ":gpu_sqrt_kernels",
        ":gpu_square_kernels",
        ":gpu_tan_kernels",
        ":gpu_tanh_kernels",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "cpu_cwise_unary_op",
    srcs = [
        "cpu_op_abs.cc",
        "cpu_op_rsqrt.cc",
        "cpu_op_sqrt.cc",
        "cpu_op_square.cc",
    ],
    tags = ["manual"],
    # Technically we only need to depend on the kernel libraries for the
    # kernels which are enabled by default. But this would make our BUILD
    # target structure uglier. We already need to make sure that those
    # targets can be built, so it should not hurt to link them in even if
    # they are currently not needed yet.
    deps = [
        ":base_cpu_op",
        ":cpu_abs_kernels",
        ":cpu_rsqrt_kernels",
        ":cpu_sqrt_kernels",
        ":cpu_square_kernels",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "gpu_cwise_binary_op",
    srcs = [":binary_gpu_kernel_srcs"],
    tags = [
        "manual",
    ],
    # Technically we only need to depend on the kernel libraries for the
    # kernels which are enabled by default. But this would make our BUILD
    # target structure uglier. We already need to make sure that those
    # targets can be built, so it should not hurt to link them in even if
    # they are currently not needed yet.
    deps = [
        ":base_gpu_op",
        ":gpu_add_v2_kernels",
        ":gpu_atan2_kernels",
        ":gpu_bitwise_and_kernels",
        ":gpu_bitwise_or_kernels",
        ":gpu_bitwise_xor_kernels",
        ":gpu_complex_kernels",
        ":gpu_div_kernels",
        ":gpu_equal_kernels",
        ":gpu_floor_div_kernels",
        ":gpu_greater_equal_kernels",
        ":gpu_greater_kernels",
        ":gpu_left_shift_kernels",
        ":gpu_less_equal_kernels",
        ":gpu_less_kernels",
        ":gpu_logical_and_kernels",
        ":gpu_logical_or_kernels",
        ":gpu_maximum_kernels",
        ":gpu_minimum_kernels",
        ":gpu_mul_kernels",
        ":gpu_not_equal_kernels",
        ":gpu_polygamma_kernels",
        ":gpu_pow_kernels",
        ":gpu_right_shift_kernels",
        ":gpu_squared_difference_kernels",
        ":gpu_sub_kernels",
        ":gpu_zeta_kernels",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "cpu_cwise_binary_op",
    srcs = [
        "cpu_op_add.cc",
    ],
    tags = ["manual"],
    deps = [
        ":base_cpu_op",
        ":cpu_add_v2_kernels",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "cwise_op",
    srcs = [],
    tags = ["no_rocm"],
    # Technically these libraries don't need --config=cuda or --config=rocm,
    # but we want to avoid building them if they are not needed.
    deps = if_cuda_or_rocm([
        ":gpu_cwise_unary_op",
        ":gpu_cwise_binary_op",
    ]) + if_mlir_experimental_kernels_enabled([":experimental_cwise_op"]),
)

tf_kernel_library(
    name = "experimental_cwise_op",
    srcs = [],
    deps = [
        ":cpu_cwise_binary_op",
        ":cpu_cwise_unary_op",
    ],
)

tf_kernel_library(
    name = "cast_op",
    srcs = ["gpu_op_cast.cc"],
    tags = [
        "manual",
    ],
    deps = [
        ":base_gpu_op",
        ":gpu_cast_kernels",
        "//third_party/eigen3",
    ],
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
    size = "small",
    srcs = if_mlir_generated_gpu_kernels_enabled(["gpu_unary_ops_test.cc"]),
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171341759): re-enable.
    ],
    deps = [
        ":base_ops_test",
        ":base_unary_ops_test",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
    ],
)

tf_cc_test(
    name = "cpu_unary_ops_test",
    size = "small",
    srcs = if_mlir_generated_gpu_kernels_enabled(["cpu_unary_ops_test.cc"]),
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
    srcs = if_mlir_generated_gpu_kernels_enabled(["gpu_binary_ops_test.cc"]),
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # b/173033461
    ],
    deps = [
        ":base_binary_ops_test",
        ":base_ops_test",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
    ],
)

tf_cuda_cc_test(
    name = "cpu_binary_ops_test",
    size = "medium",
    srcs = if_mlir_generated_gpu_kernels_enabled(["cpu_binary_ops_test.cc"]),
    deps = [
        ":base_binary_ops_test",
        ":base_ops_test",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
    ],
)

# TODO(b/160731748): Re-enable when it works again.
# gpu_kernel_library(
#     name = "gpu_bias_add_lib",
#     op = "bias_add",
#     tile_size = "16x16",
#     types = [
#         "f16",
#         "f32",
#         "f64",
#     ],
# )

# TODO(b/160190568): Re-enable when it works again.
# gpu_kernel_library(
#     name = "gpu_relu_lib",
#     op = "relu",
#     tile_size = "256",
#     types = [
#         "f16",
#         "f32",
#         "f64",
#     ],
# )

# TODO(b/25387198): Add an int32 kernel.
gpu_kernel_library(
    name = "gpu_abs_lib",
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
    name = "gpu_acos_lib",
    op = "acos",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    # Cannot vectorize.
)

gpu_kernel_library(
    name = "gpu_acosh_lib",
    op = "acosh",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    # May be compute-bound.
    # unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_angle_lib",
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
    name = "gpu_asin_lib",
    op = "asin",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    # Cannot vectorize.
)

gpu_kernel_library(
    name = "gpu_asinh_lib",
    op = "asinh",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    # Cannot vectorize.
)

gpu_kernel_library(
    name = "gpu_atan_lib",
    op = "atan",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_atanh_lib",
    op = "atanh",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_conj_lib",
    op = "conj",
    tile_size = "256",
    types = [
        "c64",
        "c128",
    ],
    unroll_factors = "2",
)

gpu_kernel_library(
    name = "gpu_cosh_lib",
    op = "cosh",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    # May be compute-bound.
    # unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_erf_lib",
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
    name = "gpu_erfc_lib",
    op = "erfc",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_imag_lib",
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
    name = "gpu_invert_lib",
    op = "invert",
    tile_size = "256",
    types = [
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_logical_not_lib",
    op = "logical_not",
    tile_size = "256",
    types = ["i1"],
)

gpu_kernel_library(
    name = "gpu_real_lib",
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

gpu_kernel_library(
    name = "gpu_polygamma_lib",
    op = "polygamma",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
)

gpu_kernel_library(
    name = "gpu_sign_lib",
    op = "sign",
    tile_size = "256",
    types = [
        # TODO(b/162577610): Add bf16, c64 and c128.
        "f16",
        "f32",
        "f64",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_sinh_lib",
    op = "sinh",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    # May be compute-bound.
    # unroll_factors = "4",
)

[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
        op = op,
        tile_size = "1024",
        types = [
            "f16",
            "f32",
            "f64",
            "i64",
        ],
        unroll_factors = "4",
    )
    for op in [
        "square",
        "add_v2",
        "squared_difference",
        "sub",
    ]
]

gpu_kernel_library(
    name = "gpu_complex_lib",
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
    name = "gpu_complex_abs_lib",
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
    name = "gpu_div_lib",
    op = "div",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i16",
        "i64",
    ],
)

gpu_kernel_library(
    name = "gpu_mul_lib",
    op = "mul",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i64",
    ],
    unroll_factors = "4",
)

# Bitwise operations.
[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
        op = op,
        tile_size = "1024",
        types = [
            "i8",
            "i16",
            "i32",
            "i64",
            # TODO(b/172804967): Enable once fixed.
            # "ui8",
            # "ui16",
            # "ui32",
            # "ui64",
        ],
        unroll_factors = "4",
    )
    for op in [
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "left_shift",
        "right_shift",
    ]
]

gpu_kernel_library(
    name = "gpu_atan2_lib",
    op = "atan2",
    tile_size = "256,1,1",
    types = [
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

# Logical operations.
[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
        op = op,
        tile_size = "1024",
        types = [
            "i1",
        ],
    )
    for op in [
        "logical_and",
        "logical_or",
    ]
]

[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
        op = op,
        output_types = ["i1"] * 8,
        tile_size = "1024",
        types = [
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
    for op in [
        "equal",
        "not_equal",
    ]
]

[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
        op = op,
        output_types = ["i1"] * 7,
        tile_size = "1024",
        types = [
            "f16",
            "f32",
            "f64",
            "i8",
            "i16",
            "i32",
            "i64",
        ],
        unroll_factors = "4",
    )
    for op in [
        "less",
        "less_equal",
        "greater",
        "greater_equal",
    ]
]

[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
        op = op,
        tile_size = "1024",
        types = [
            "f16",
            "f32",
            "f64",
            "i16",
            "i32",
            "i64",
        ],
        unroll_factors = "4",
    )
    for op in [
        "maximum",
        "minimum",
    ]
]

# Kernels that support all floating-point and signed int types.
[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
        op = op,
        tile_size = "256",
        types = [
            "f16",
            "f32",
            "f64",
            "i8",
            "i16",
            "i32",
            "i64",
        ],
        unroll_factors = "4",
    )
    for op in [
        "neg",
    ]
]

gpu_kernel_library(
    name = "gpu_floor_div_lib",
    op = "floor_div",
    tile_size = "1024",
    # TODO(172804967): Enable for integer types also once unsigned integers are
    # supported.
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

# Kernels that support all floating-point types.
[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
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
        "ceil",
        "digamma",
        "exp",
        "expm1",
        "floor",
        "lgamma",
        "log",
        "log1p",
        "rsqrt",
        "sqrt",
        "tanh",
    ]
]

# Kernels that support all floating-point types but have i1 output.
[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
        op = op,
        output_types = ["i1"] * 3,
        tile_size = "256",
        types = [
            "f16",
            "f32",
            "f64",
        ],
        unroll_factors = "4",
    )
    for op in [
        "is_finite",
        "is_inf",
        "is_nan",
    ]
]

# Kernels that support all floating-point types but cannot be vectorized.
[
    gpu_kernel_library(
        name = "gpu_" + op + "_lib",
        op = op,
        tile_size = "256",
        types = [
            "f16",
            "f32",
            "f64",
        ],
    )
    for op in [
        "cos",
        "sin",
        "tan",
    ]
]

gpu_kernel_library(
    name = "gpu_cast_lib",
    op = "cast",
    # We generate all combinations of input types/output types from the set
    # {i1, i8, i16, i32, i64, f16, f32, f64}. The easiest way to do this is to
    # repeat each input type 8 times and match it with the 8 different output
    # types (thus, the list of 8 different output types needs to be repeated 8
    # times as well).
    output_types = [
        "i1",
        "i8",
        "i16",
        "i32",
        "i64",
        "f16",
        "f32",
        "f64",
    ] * 8,
    tile_size = "256",
    types = ["i1"] * 8 + ["i8"] * 8 + ["i16"] * 8 + ["i32"] * 8 + ["i64"] * 8 + ["f16"] * 8 + ["f32"] * 8 + ["f64"] * 8,
    unroll_factors = "4",
)

gpu_kernel_library(
    name = "gpu_pow_lib",
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
    # The zeta kernels needs many registers so tile at 256.
    name = "gpu_zeta_lib",
    op = "zeta",
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    # TODO(b/178388085): Enable unrolling after vectorization is fixed.
    # unroll_factors = "4",
)

cpu_kernel_library(
    name = "cpu_abs_lib",
    op = "abs",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

cpu_kernel_library(
    name = "cpu_rsqrt_lib",
    op = "rsqrt",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

cpu_kernel_library(
    name = "cpu_sqrt_lib",
    op = "sqrt",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

cpu_kernel_library(
    name = "cpu_square_lib",
    op = "square",
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

cpu_kernel_library(
    name = "cpu_add_v2_lib",
    op = "add_v2",
    tile_size = "1024",
    types = [
        "f16",
        "f32",
        "f64",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)
