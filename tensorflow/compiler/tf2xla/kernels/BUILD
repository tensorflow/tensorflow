licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//tensorflow/compiler/tf2xla:internal"],
)

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow/compiler/xla:xla.bzl", "export_dynamic_linkopts")

tf_kernel_library(
    name = "xla_ops",
    srcs = [
        "aggregate_ops.cc",
        "arg_op.cc",
        "batch_matmul_op.cc",
        "batch_norm_op.cc",
        "batchtospace_op.cc",
        "bcast_ops.cc",
        "bias_ops.cc",
        "binary_ops.cc",
        "cast_op.cc",
        "concat_op.cc",
        "const_op.cc",
        "conv_ops.cc",
        "cross_op.cc",
        "cwise_ops.cc",
        "diag_op.cc",
        "dynamic_stitch_op.cc",
        "elu_op.cc",
        "fill_op.cc",
        "function_ops.cc",
        "gather_op.cc",
        "identity_op.cc",
        "is_finite_op.cc",
        "l2loss_op.cc",
        "lrn_ops.cc",
        "matmul_op.cc",
        "mirror_pad_op.cc",
        "no_op.cc",
        "one_hot_op.cc",
        "pack_op.cc",
        "pad_op.cc",
        "pooling_ops.cc",
        "random_ops.cc",
        "reduction_ops.cc",
        "reduction_ops_common.cc",
        "relu_op.cc",
        "reshape_op.cc",
        "retval_op.cc",
        "reverse_op.cc",
        "segment_reduction_ops.cc",
        "select_op.cc",
        "sendrecv_ops.cc",
        "sequence_ops.cc",
        "shape_op.cc",
        "slice_op.cc",
        "softmax_op.cc",
        "spacetobatch_op.cc",
        "split_op.cc",
        "stack_ops.cc",
        "strided_slice_op.cc",
        "tensor_array_ops.cc",
        "tile_ops.cc",
        "training_ops.cc",
        "transpose_op.cc",
        "unary_ops.cc",
        "unpack_op.cc",
        "variable_ops.cc",
    ],
    hdrs = [
        "cwise_ops.h",
        "gather_op.h",
        "gather_op_helpers.h",
        "reduction_ops.h",
    ],
    deps = [
        ":while_op",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/ops:sendrecv_ops",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow_opensource",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/kernels:concat_lib",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:no_op",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:pooling_ops",
        "//tensorflow/core/kernels:sendrecv_ops",
        "//tensorflow/core/kernels:transpose_op",
    ],
)

tf_kernel_library(
    name = "while_op",
    srcs = ["while_op.cc"],
    hdrs = ["while_op.h"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/ops:functional_ops",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow_opensource",
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
        "//tensorflow/core/kernels:bounds_check",
    ],
)

tf_kernel_library(
    name = "gather_op_kernel_float_int32",
    srcs = ["gather_op_kernel_float_int32.cc"],
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
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "index_ops_kernel_argmax_float_2d",
    srcs = ["index_ops_kernel_argmax_float_2d.cc"],
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
