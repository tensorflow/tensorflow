# Generates CUDA kernels using MLIR codegen.

load(
    "//tensorflow/core/kernels/mlir_generated:build_defs.bzl",
    "gen_kernel_library",
    "if_mlir_generated_gpu_kernels_enabled",
    "if_mlir_unranked_kernels_enabled",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
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

config_setting(
    name = "mlir_generated_gpu_kernels_disabled",
    define_values = {
        "tensorflow_enable_mlir_generated_gpu_kernels": "0",
    },
)

config_setting(
    name = "mlir_use_unranked_kernels",
    define_values = {"enable_unranked_kernels": "1"},
)

filegroup(
    name = "enabled_unary_unranked_kernel_srcs",
    srcs = [
        "unranked_op_gpu_abs.cc",
        "unranked_op_gpu_tanh.cc",
    ],
    compatible_with = get_compatible_with_cloud(),
)

filegroup(
    name = "experimental_unary_unranked_kernel_srcs",
    srcs = [
        "unranked_op_gpu_ceil.cc",
        "unranked_op_gpu_conj.cc",
        "unranked_op_gpu_cos.cc",
        "unranked_op_gpu_exp.cc",
        "unranked_op_gpu_floor.cc",
        "unranked_op_gpu_imag.cc",
        "unranked_op_gpu_is_inf.cc",
        "unranked_op_gpu_log.cc",
        "unranked_op_gpu_logical_not.cc",
        "unranked_op_gpu_real.cc",
        "unranked_op_gpu_rsqrt.cc",
        "unranked_op_gpu_sign.cc",
        "unranked_op_gpu_sin.cc",
        "unranked_op_gpu_sqrt.cc",
    ],
    compatible_with = get_compatible_with_cloud(),
)

filegroup(
    name = "unary_unranked_kernel_srcs",
    srcs = [
        ":enabled_unary_unranked_kernel_srcs",
    ] + if_mlir_unranked_kernels_enabled(
        if_true = [":experimental_unary_unranked_kernel_srcs"],
    ),
    compatible_with = get_compatible_with_cloud(),
)

filegroup(
    name = "unary_kernel_srcs",
    srcs = if_mlir_unranked_kernels_enabled(
        if_false = [
            "cwise_op_gpu_abs.cc",
            "cwise_op_gpu_base.cc",
            "cwise_op_gpu_base.h",
            "cwise_op_gpu_tanh.cc",
        ],
        if_true = [":unary_unranked_kernel_srcs"],
    ),
    compatible_with = get_compatible_with_cloud(),
)

cc_library(
    name = "unranked_op_gpu_base",
    srcs = ["unranked_op_gpu_base.cc"],
    hdrs = ["unranked_op_gpu_base.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/tools/kernel_gen:tf_framework_c_interface",
        "//tensorflow/compiler/mlir/tools/kernel_gen:tf_gpu_runtime_wrappers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:allocation_description_proto_cc",
        "//tensorflow/core/framework:op_requires",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:mlir_c_runner_utils",
    ],
)

tf_kernel_library(
    name = "cwise_unary_op",
    srcs = [":unary_kernel_srcs"],
    tags = [
        "manual",
    ],
    deps = if_mlir_unranked_kernels_enabled(
        if_false = [
            ":abs_kernels",
            ":tanh_kernels",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/synchronization",
            "@com_google_absl//absl/types:span",
            "//third_party/eigen3",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core/platform:stream_executor",
        ],
        if_true = [
            # Technically we only need to depend on the kernel libraries for the
            # unranked kernels which are enabled by default. But this would
            # make our BUILD target structure uglier. We already need to make
            # sure that those targets can be built, so it should not hurt to
            # link them in even if they are currently not needed yet.
            ":abs_unranked_kernels",
            ":ceil_unranked_kernels",
            ":conj_unranked_kernels",
            ":cos_unranked_kernels",
            ":exp_unranked_kernels",
            ":floor_unranked_kernels",
            ":imag_unranked_kernels",
            ":is_inf_unranked_kernels",
            ":log_unranked_kernels",
            ":logical_not_unranked_kernels",
            ":real_unranked_kernels",
            ":rsqrt_unranked_kernels",
            ":sign_unranked_kernels",
            ":sin_unranked_kernels",
            ":sqrt_unranked_kernels",
            ":tanh_unranked_kernels",
            ":unranked_op_gpu_base",
            "//third_party/eigen3",
        ],
    ),
)

tf_kernel_library(
    name = "cwise_binary_op",
    srcs = [
        "unranked_op_gpu_add.cc",
        "unranked_op_gpu_bitwise_and.cc",
        "unranked_op_gpu_bitwise_or.cc",
        "unranked_op_gpu_bitwise_xor.cc",
        "unranked_op_gpu_equal.cc",
        "unranked_op_gpu_logical_and.cc",
        "unranked_op_gpu_logical_or.cc",
    ],
    tags = [
        "manual",
    ],
    deps = [
        ":add_v2_unranked_kernels",
        ":bitwise_and_unranked_kernels",
        ":bitwise_or_unranked_kernels",
        ":bitwise_xor_unranked_kernels",
        ":equal_unranked_kernels",
        ":greater_equal_unranked_kernels",
        ":greater_unranked_kernels",
        ":less_equal_unranked_kernels",
        ":less_unranked_kernels",
        ":logical_and_unranked_kernels",
        ":logical_or_unranked_kernels",
        ":maximum_unranked_kernels",
        ":minimum_unranked_kernels",
        ":not_equal_unranked_kernels",
        ":unranked_op_gpu_base",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "cwise_op",
    srcs = [],
    tags = [
        "no_rocm",
    ],
    # Technically these libraries don't need --config=cuda or --config=rocm,
    # but we want to avoid building them if they are not needed.
    deps = if_cuda_or_rocm([
        ":cwise_unary_op",
    ]) + if_mlir_unranked_kernels_enabled(
        [
            ":cwise_binary_op",
        ],
    ),
)

tf_cuda_cc_test(
    name = "gpu_unary_ops_test",
    size = "small",
    srcs = if_mlir_generated_gpu_kernels_enabled(["gpu_unary_ops_test.cc"]),
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # TODO(b/171341759): re-enable.
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

tf_cuda_cc_test(
    name = "gpu_binary_ops_test",
    size = "small",
    srcs = if_mlir_generated_gpu_kernels_enabled(["gpu_binary_ops_test.cc"]),
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # b/173033461
    ],
    deps = [
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
    ],
)

# TODO(b/160731748): Re-enable when it works again.
# gen_kernel_library(
#     name = "bias_add",
#     tile_size = "16x16",
#     types = [
#         "f16",
#         "f32",
#         "f64",
#     ],
# )

# TODO(b/160190568): Re-enable when it works again.
# gen_kernel_library(
#     name = "relu",
#     tile_size = "256",
#     types = [
#         "f16",
#         "f32",
#         "f64",
#     ],
# )

gen_kernel_library(
    name = "abs",
    generate_unranked = True,
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "conj",
    generate_unranked = True,
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    unroll_factors = "2",
)

gen_kernel_library(
    name = "imag",
    generate_unranked = True,
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    unroll_factors = "1",
)

gen_kernel_library(
    name = "invert",
    generate_unranked = True,
    tile_size = "256",
    types = [
        "i8",
        "i16",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "is_inf",
    generate_ranked = False,
    generate_unranked = True,
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "logical_not",
    generate_unranked = True,
    tile_size = "256",
    types = ["i1"],
    unroll_factors = "1",
)

gen_kernel_library(
    name = "real",
    generate_unranked = True,
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    unroll_factors = "1",
)

gen_kernel_library(
    name = "sign",
    generate_unranked = True,
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

gen_kernel_library(
    name = "add_v2",
    generate_ranked = False,
    generate_unranked = True,
    tile_size = "256,1,1",
    types = [
        "f16",
        "f32",
        "f64",
        "i64",
    ],
    # TODO(b/174543802): Enable once fusion heuristics is better.
    # unroll_factors = "4",
)

# Bitwise operations.
[
    gen_kernel_library(
        name = name,
        generate_ranked = False,
        generate_unranked = True,
        tile_size = "256,1,1",
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
        # TODO(b/174543802): Enable once fusion heursitics is better.
        # unroll_factors = "4",
    )
    for name in [
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    ]
]

# Logical operations.
[
    gen_kernel_library(
        name = name,
        generate_ranked = False,
        generate_unranked = True,
        tile_size = "256,1,1",
        types = [
            "i1",
        ],
        # TODO(b/174543802): Enable once fusion heursitics is better.
        # unroll_factors = "4",
    )
    for name in [
        "logical_and",
        "logical_or",
    ]
]

[
    gen_kernel_library(
        name = name,
        generate_ranked = False,
        generate_unranked = True,
        tile_size = "256,1,1",
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
        # TODO(b/174543802): Enable once fusion heuristics is better.
        # unroll_factors = "4",
    )
    for name in [
        "equal",
        "not_equal",
    ]
]

[
    gen_kernel_library(
        name = name,
        generate_ranked = False,
        generate_unranked = True,
        tile_size = "256,1,1",
        types = [
            "f16",
            "f32",
            "f64",
            "i8",
            "i16",
            "i32",
            "i64",
        ],
        # TODO(b/174543802): Enable once fusion heuristics is better.
        # unroll_factors = "4",
    )
    for name in [
        "less",
        "less_equal",
        "greater",
        "greater_equal",
    ]
]

[
    gen_kernel_library(
        name = name,
        generate_ranked = False,
        generate_unranked = True,
        tile_size = "256,1,1",
        types = [
            "f16",
            "f32",
            "f64",
            "i16",
            "i32",
            "i64",
        ],
        # TODO(b/174543802): Enable once fusion heuristics is better.
        # unroll_factors = "4",
    )
    for name in [
        "maximum",
        "minimum",
    ]
]

# Kernels that support all floating-point types.
[
    gen_kernel_library(
        name = name,
        generate_unranked = True,
        tile_size = "256",
        types = [
            "f16",
            "f32",
            "f64",
        ],
        unroll_factors = "4",
    )
    for name in [
        "ceil",
        "exp",
        "floor",
        "is_finite",
        "log",
        "neg",
        "rsqrt",
        "sqrt",
        "tanh",
    ]
]

# Kernels that support all floating-point types but cannot be vectorized.
[
    gen_kernel_library(
        name = name,
        generate_unranked = True,
        tile_size = "256",
        types = [
            "f16",
            "f32",
            "f64",
        ],
        unroll_factors = "1",
    )
    for name in [
        "cos",
        "sin",
    ]
]
