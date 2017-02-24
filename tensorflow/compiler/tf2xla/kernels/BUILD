licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
    features = ["no_layering_check"],
)

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow/compiler/xla:xla.bzl", "export_dynamic_linkopts")

tf_kernel_library(
    name = "xla_ops",
    srcs = [
        "aggregate_ops.cc",
        "batch_matmul_op.cc",
        "bcast_ops.cc",
        "bias_ops.cc",
        "binary_ops.cc",
        "cast_op.cc",
        "concat_op.cc",
        "conv_ops.cc",
        "cwise_ops.cc",
        "declaration_op.cc",
        "depthwise_conv_ops.cc",
        "diag_op.cc",
        "dynamic_stitch_op.cc",
        "fill_op.cc",
        "function_ops.cc",
        "identity_op.cc",
        "l2loss_op.cc",
        "lrn_ops.cc",
        "matmul_op.cc",
        "no_op.cc",
        "pack_op.cc",
        "pad_op.cc",
        "pooling_ops.cc",
        "random_ops.cc",
        "reduction_ops.cc",
        "reduction_ops_common.cc",
        "relu_op.cc",
        "reshape_op.cc",
        "retval_op.cc",
        "select_op.cc",
        "sequence_ops.cc",
        "shape_op.cc",
        "slice_op.cc",
        "softmax_op.cc",
        "split_op.cc",
        "strided_slice_op.cc",
        "tile_ops.cc",
        "transpose_op.cc",
        "unary_ops.cc",
        "unpack_op.cc",
        "variable_ops.cc",
    ],
    hdrs = [
        "cwise_ops.h",
        "reduction_ops.h",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow_opensource",
        "//tensorflow/core/kernels:concat_lib",
        "//tensorflow/core/kernels:conv_2d",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:depthwise_conv_op",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:no_op",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:pooling_ops",
        "//tensorflow/core/kernels:sendrecv_ops",
        "//tensorflow/core/kernels:transpose_op",
    ],
)

# Kernels that only work on CPU, because they use XLA custom calls.
# Only link this when using the CPU backend for XLA.
#
# TODO(cwhipkey): move into xla_ops when ops can be registered for
# CPU compilation only (b/31363654).
tf_kernel_library(
    name = "xla_cpu_only_ops",
    srcs = [
        "gather_op.cc",
        "index_ops.cc",
    ],
    deps = [
        ":gather_op_kernel_float_int32",
        ":gather_op_kernel_float_int64",
        ":index_ops_kernel_argmax_float_1d",
        ":index_ops_kernel_argmax_float_2d",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow_opensource",
    ],
)

tf_kernel_library(
    name = "gather_op_kernel_float_int32",
    srcs = ["gather_op_kernel_float_int32.cc"],
    # Makes the custom-call function visible to LLVM during JIT.
    linkopts = export_dynamic_linkopts,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_local_runtime_context",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/kernels:gather_functor",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "gather_op_kernel_float_int64",
    srcs = ["gather_op_kernel_float_int64.cc"],
    # Makes the custom-call function visible to LLVM during JIT.
    linkopts = export_dynamic_linkopts,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_local_runtime_context",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/kernels:gather_functor",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "index_ops_kernel_argmax_float_1d",
    srcs = ["index_ops_kernel_argmax_float_1d.cc"],
    # Makes the custom-call function visible to LLVM during JIT.
    linkopts = export_dynamic_linkopts,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "index_ops_kernel_argmax_float_2d",
    srcs = ["index_ops_kernel_argmax_float_2d.cc"],
    # Makes the custom-call function visible to LLVM during JIT.
    linkopts = export_dynamic_linkopts,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

# -----------------------------------------------------------------------------

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
