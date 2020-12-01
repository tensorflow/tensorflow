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
    name = "unary_kernel_srcs",
    srcs = if_mlir_unranked_kernels_enabled(
        if_false = [
            "cwise_op_gpu_abs.cc",
            "cwise_op_gpu_base.cc",
            "cwise_op_gpu_base.h",
            "cwise_op_gpu_tanh.cc",
        ],
        if_true = [
            "unranked_op_gpu_abs.cc",
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
            "unranked_op_gpu_tanh.cc",
        ],
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
    srcs = ["unranked_gpu_add.cc"],
    tags = [
        "manual",
    ],
    deps = [
        ":add_v2_unranked_kernels",
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
    name = "gpu_add_test",
    size = "small",
    srcs = if_mlir_generated_gpu_kernels_enabled(["gpu_add_test.cc"]),
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
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:ops_testutil",
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
    unroll_factors = "4",
)

gen_kernel_library(
    name = "imag",
    generate_unranked = True,
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    unroll_factors = "4",
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
    unroll_factors = "4",
)

gen_kernel_library(
    name = "real",
    generate_unranked = True,
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
    unroll_factors = "4",
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
    # TODO(b/174543802): Enable once fusion heursitics is better.
    # unroll_factors = "4",
)

gen_kernel_library(
    name = "equal",
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
    # TODO(b/174543802): Enable once fusion heursitics is better.
    # unroll_factors = "4",
)

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
        "cos",
        "exp",
        "floor",
        "is_finite",
        "log",
        "neg",
        "rsqrt",
        "sin",
        "sqrt",
        "tanh",
    ]
]
