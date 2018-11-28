# Description:
# Op kernel implementations for TensorFlow.
#
# Note: Any test that uses GPU support and which we would like to
# benchmark should be linked statically so that it can be executed
# from a py_binary or cuda_py_test test logger.  For such a test,
# append "_gpu" to the test name to invoke the GPU benchmarks.  Example:
#
#   # for CPU tests
#   $ bazel test --config opt //third_party/tensorflow/core/kernels:my_op_test
#   # for GPU benchmarks
#   $ bazel run --config opt --config=cuda //third_party/tensorflow/core/kernels:my_op_test_gpu -- --benchmarks=..
#
package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

package_group(
    name = "friends",
    packages = [
        "//learning/brain/contrib/...",
        "//learning/brain/research/sparse_matrix/...",
        "//learning/faster_training/...",
        "//tensorflow/...",
        "//tensorflow_text/...",
        "//third_party/car/...",
    ],
)

load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_cc_tests",
    "tf_cc_binary",
    "tf_copts",
    "tf_cuda_library",
    "tf_opts_nortti_if_android",
    "tf_kernel_library",
    "tf_mkl_kernel_library",
    "cc_header_only_library",
    "if_not_windows",
)
load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_tests")
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
    "if_mkl_ml",
    "mkl_deps",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

config_setting(
    # Add "--define tensorflow_xsmm=1" to your build command to use libxsmm for
    # sparse matrix multiplications. You will also need appropriate -mavx*
    # options, as required by specific op you use.
    name = "xsmm",
    values = {
        "define": "tensorflow_xsmm=1",
    },
)

config_setting(
    # Add "--define tensorflow_xsmm_convolutions=1" to your build command to
    # use libxsmm for forward convolutions. You will also need appropriate
    # -mavx* # options, as required by specific op you use.
    name = "xsmm_convolutions",
    values = {
        "define": "tensorflow_xsmm_convolutions=1",
    },
)

config_setting(
    # Add "--define tensorflow_xsmm_convolutions=1 --define
    # tensorflow_xsmm_backward_convolutions=1" to your build command to use libxsmm for
    # backward convolutions (and possibly more in the future). You will also
    # need appropriate -mavx* options, as required by specific op you use.
    name = "xsmm_backward_convolutions",
    values = {
        "define": "tensorflow_xsmm_backward_convolutions=1",
    },
)

config_setting(
    # Add "--define tensorflow_mkldnn_contraction_kernel=1" to your build command to use mkldnn
    # sgemm in Eigen tensor contractions (matrix multiplications and convolutions). The mkldnn
    # kernels are generated at runtime and use avx/avx2/fma/avx512 based on cpu status registers
    # (https://en.wikipedia.org/wiki/CPUID).
    name = "mkldnn_contraction_kernel",
    values = {
        "define": "tensorflow_mkldnn_contraction_kernel=1",
    },
)

# Public support libraries ----------------------------------------------------

cc_library(
    name = "assign_op",
    hdrs = ["assign_op.h"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "strided_slice_op",
    srcs = [
        "strided_slice_op.cc",
        "strided_slice_op_inst_0.cc",
        "strided_slice_op_inst_1.cc",
        "strided_slice_op_inst_2.cc",
        "strided_slice_op_inst_3.cc",
        "strided_slice_op_inst_4.cc",
        "strided_slice_op_inst_5.cc",
        "strided_slice_op_inst_6.cc",
        "strided_slice_op_inst_7.cc",
    ],
    hdrs = [
        "slice_op.h",
        "strided_slice_op.h",
        "strided_slice_op_impl.h",
    ],
    gpu_srcs = [
        "slice_op.h",
        "strided_slice_op.h",
        "strided_slice_op_impl.h",
        "strided_slice_op_gpu.cu.cc",
    ],
    deps = [
        ":bounds_check",
        ":dense_update_functor",
        ":ops_util",
        ":training_op_helpers",
        ":variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "collective_ops",
    prefix = "collective_ops",
    deps = [
        "//tensorflow/core:collective_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "concat_lib",
    srcs = [
        "concat_lib_cpu.cc",
        "concat_lib_gpu.cc",
    ],
    hdrs = [
        "concat_lib.h",
        "concat_lib_cpu.h",
    ],
    gpu_srcs = [
        "concat_lib_gpu_impl.cu.cc",
        "concat_lib.h",
        "cuda_device_array.h",
        "cuda_device_array_gpu.h",
    ],
    deps = [
        ":bounds_check",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
    alwayslink = 0,
)

cc_library(
    name = "concat_lib_hdrs",
    hdrs = [
        "concat_lib.h",
        "concat_lib_cpu.h",
    ],
    deps = ["//third_party/eigen3"],
)

tf_kernel_library(
    name = "conv_2d",
    hdrs = ["conv_2d.h"],
    gpu_srcs = [
        "conv_2d_gpu.cu.cc",
        "conv_2d.h",
    ],
    deps = [
        ":eigen_helpers",
        ":fill_functor",
        ":gpu_util_hdrs",
        ":image_resizer_state",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_ops_op_lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "conv_2d_hdrs",
    hdrs = ["conv_2d.h"],
    deps = [
        ":eigen_helpers",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "extract_image_patches_op",
    prefix = "extract_image_patches_op",
    deps = [
        ":bounds_check",
        ":eigen_helpers",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "extract_volume_patches_op",
    prefix = "extract_volume_patches_op",
    deps = [
        ":bounds_check",
        ":eigen_helpers",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "conv_3d",
    hdrs = ["conv_3d.h"],
    deps = [
        ":eigen_helpers",
        "//tensorflow/core:framework",
    ],
)

tf_kernel_library(
    name = "fill_functor",
    prefix = "fill_functor",
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "initializable_lookup_table",
    srcs = ["initializable_lookup_table.cc"],
    hdrs = ["initializable_lookup_table.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "lookup_util",
    srcs = ["lookup_util.cc"],
    hdrs = ["lookup_util.h"],
    deps = [
        ":initializable_lookup_table",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "nccl_kernels",
    srcs = if_cuda([
        "nccl_ops.cc",
    ]),
    deps = if_cuda([
        "@local_config_nccl//:nccl",
        "//tensorflow/core/nccl:nccl_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:nccl_ops_op_lib",
    ]),
)

tf_cuda_library(
    name = "ops_testutil",
    testonly = 1,
    srcs = ["ops_testutil.cc"],
    hdrs = ["ops_testutil.h"],
    cuda_deps = [
        "//tensorflow/core:gpu_lib",
        "//tensorflow/core:gpu_runtime",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "ops_util",
    hdrs = ["ops_util.h"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    deps = [
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "ops_util_hdrs",
    hdrs = ["ops_util.h"],
    deps = ["//third_party/eigen3"],
)

cc_library(
    name = "conv_ops_gpu_hdrs",
    hdrs = ["conv_ops_gpu.h"],
)

cc_library(
    name = "gpu_util_hdrs",
    hdrs = ["gpu_utils.h"],
)

tf_cc_test(
    name = "ops_util_test",
    size = "small",
    srcs = ["ops_util_test.cc"],
    deps = [
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "reshape_util",
    srcs = ["reshape_util.cc"],
    hdrs = ["reshape_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "variable_ops_test",
    size = "small",
    srcs = ["variable_ops_test.cc"],
    deps = [
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_kernel_library(
    name = "stage_op",
    srcs = ["stage_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "map_stage_op",
    srcs = ["map_stage_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "queue_base",
    srcs = ["queue_base.cc"],
    hdrs = ["queue_base.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "queue_op",
    srcs = ["queue_op.cc"],
    hdrs = ["queue_op.h"],
    deps = [
        ":queue_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "priority_queue",
    srcs = ["priority_queue.cc"],
    hdrs = ["priority_queue.h"],
    deps = [
        ":queue_base",
        ":typed_queue",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "batch_kernels",
    srcs = ["batch_kernels.cc"],
    deps = [
        "//tensorflow/core:batch_ops_op_lib",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:concat_lib_hdrs",
        "//tensorflow/core/kernels:ops_util_hdrs",
        "//tensorflow/core/kernels:split_lib_hdrs",
        "//tensorflow/core/kernels/batching_util:periodic_function_dynamic",
        "//tensorflow/core/kernels/batching_util:shared_batch_scheduler_hdrs",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "record_input_op",
    srcs = [
        "record_input_op.cc",
        "record_yielder.cc",
        "record_yielder.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "save_restore_tensor",
    srcs = ["save_restore_tensor.cc"],
    hdrs = ["save_restore_tensor.h"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    deps = [
        ":bounds_check",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

tf_kernel_library(
    name = "split_lib",
    srcs = ["split_lib_cpu.cc"],
    hdrs = ["split_lib.h"],
    gpu_srcs = [
        "split_lib_gpu.cu.cc",
        "split_lib.h",
    ],
    deps = [
        ":cuda_device_array",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
    alwayslink = 0,
)

cc_library(
    name = "split_lib_hdrs",
    hdrs = [
        "split_lib.h",
    ],
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "typed_queue",
    hdrs = ["typed_queue.h"],
    deps = [
        ":queue_base",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "training_op_helpers",
    srcs = ["training_op_helpers.cc"],
    hdrs = ["training_op_helpers.h"],
    visibility = [":friends"],
    deps = [
        ":dense_update_functor",
        ":variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "bounds_check",
    hdrs = ["bounds_check.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework_bounds_check",
    ],
)

# Private support libraries ---------------------------------------------------

cc_header_only_library(
    name = "bounds_check_lib",
    deps = [":bounds_check"],
)

cc_library(
    name = "cuda_device_array",
    hdrs = [
        "cuda_device_array.h",
        "cuda_device_array_gpu.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
    ],
)

# Depending on a build configuration this target provides custom kernel for Eigen
# tensor contractions (small matrix multiplication kernel used to multiple together
# blocks of the original tensors).
#
# 0) Default contraction kernel is Eigen::internal::gebp_kernel.
#
# 1) --define tensorflow_mkldnn_contraction_kernel=1
#    Use Mkldnn single threaded sgemm. The mkldnn kernels are generated at runtime and
#    use avx/avx2/fma/avx512 based on cpu status registers (https://en.wikipedia.org/wiki/CPUID).
#
# If you use `tensor.contract(other_tensor)` in your code, you must include additional header
# to get the benefit of custom contraction kernel:
#
#   #if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#   #include "third_party/tensorflow/core/kernels/eigen_contraction_kernel.h"
#   #endif
cc_library(
    name = "eigen_contraction_kernel",
    hdrs = ["eigen_contraction_kernel.h"],
    defines = select({
        ":mkldnn_contraction_kernel": [
            "TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL",
            "TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//third_party/eigen3",
    ] + select({
        ":mkldnn_contraction_kernel": ["//third_party/intel_mkl_dnn:mkldnn_single_threaded"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "eigen_helpers",
    hdrs = [
        "eigen_activations.h",
        "eigen_attention.h",
        "eigen_backward_cuboid_convolutions.h",
        "eigen_backward_spatial_convolutions.h",
        "eigen_cuboid_convolution.h",
        "eigen_pooling.h",
        "eigen_softmax.h",
        "eigen_spatial_convolutions.h",
        "eigen_volume_patch.h",
    ],
    deps = [
        ":eigen_contraction_kernel",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "image_resizer_state",
    hdrs = ["image_resizer_state.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":bounds_check",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

cc_header_only_library(
    name = "image_resizer_state_lib",
    deps = [":image_resizer_state"],
)

# OpKernel libraries ----------------------------------------------------------

ARRAY_DEPS = [
    ":bounds_check",
    ":concat_lib",
    ":fill_functor",
    ":gather_functor",
    ":ops_util",
    ":transpose_functor",
    "//tensorflow/core:array_grad",
    "//tensorflow/core:array_ops_op_lib",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:proto_text",
    "//tensorflow/core:protos_all_cc",
    "//third_party/eigen3",
] + if_sycl(["//tensorflow/core:sycl_runtime"])

cc_library(
    name = "array_not_windows",
    deps = [
        ":immutable_constant_op",
    ],
)

tf_kernel_library(
    name = "immutable_constant_op",
    prefix = "immutable_constant_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "set_kernels",
    prefix = "set_kernels",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:set_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "debug_ops",
    prefix = "debug_ops",
    deps = ARRAY_DEPS + [
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core/debug:debug_io_utils",
    ],
)

cc_library(
    name = "array",
    deps = [
        ":batch_space_ops",
        ":bcast_ops",
        ":bitcast_op",
        ":broadcast_to_op",
        ":concat_op",
        ":constant_op",
        ":depth_space_ops",
        ":diag_op",
        ":edit_distance_op",
        ":extract_image_patches_op",
        ":extract_volume_patches_op",
        ":gather_nd_op",
        ":gather_op",
        ":guarantee_const_op",
        ":host_constant_op",
        ":identity_n_op",
        ":identity_op",
        ":inplace_ops",
        ":listdiff_op",
        ":matrix_band_part_op",
        ":matrix_diag_op",
        ":matrix_set_diag_op",
        ":mirror_pad_op",
        ":one_hot_op",
        ":pack_op",
        ":pad_op",
        ":quantize_and_dequantize_op",
        ":reshape_op",
        ":reverse_op",
        ":reverse_sequence_op",
        ":searchsorted_op",
        ":shape_ops",
        ":slice_op",
        ":snapshot_op",
        ":split_op",
        ":split_v_op",
        ":strided_slice_op",
        ":tile_ops",
        ":transpose_op",
        ":unique_op",
        ":unpack_op",
        ":unravel_index_op",
        ":where_op",
    ],
)

tf_kernel_library(
    name = "bcast_ops",
    prefix = "bcast_ops",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "bitcast_op",
    prefix = "bitcast_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "broadcast_to_op",
    prefix = "broadcast_to_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "concat_op",
    prefix = "concat_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "guarantee_const_op",
    prefix = "guarantee_const_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "constant_op",
    prefix = "constant_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "host_constant_op",
    prefix = "host_constant_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "diag_op",
    prefix = "diag_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "edit_distance_op",
    prefix = "edit_distance_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "gather_nd_op",
    prefix = "gather_nd_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "gather_op",
    prefix = "gather_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "identity_op",
    prefix = "identity_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "identity_n_op",
    prefix = "identity_n_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "listdiff_op",
    prefix = "listdiff_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "matrix_band_part_op",
    prefix = "matrix_band_part_op",
    deps = if_cuda([
        ":cuda_solvers",
    ]) + ARRAY_DEPS,
)

tf_kernel_library(
    name = "matrix_diag_op",
    prefix = "matrix_diag_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "matrix_set_diag_op",
    prefix = "matrix_set_diag_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "mirror_pad_op",
    prefix = "mirror_pad_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "one_hot_op",
    prefix = "one_hot_op",
    deps = ARRAY_DEPS + ["//tensorflow/core:overflow"],
)

tf_kernel_library(
    name = "pack_op",
    prefix = "pack_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "pad_op",
    prefix = "pad_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "quantize_and_dequantize_op",
    prefix = "quantize_and_dequantize_op",
    deps = ARRAY_DEPS + [":cwise_op"],
)

tf_kernel_library(
    name = "compare_and_bitpack_op",
    srcs = ["compare_and_bitpack_op.cc"],
    hdrs = ["compare_and_bitpack_op.h"],
    gpu_srcs = [
        "compare_and_bitpack_op.h",
        "compare_and_bitpack_op_gpu.cu.cc",
    ],
    deps = ARRAY_DEPS,
)

# TODO(ebrevdo): Add benchmarks once the op is in the autogen array namespace.
# tf_cuda_cc_test(
#     name = "compare_and_bitpack_op_test",
#     srcs = ["compare_and_bitpack_op_test.cc"],
#     deps = [
#         ":array",
#         ":ops_testutil",
#         ":ops_util",
#         "//third_party/eigen3",
#         "//tensorflow/cc:cc_ops",
#         "//tensorflow/cc:cc_ops_internal",
#         "//tensorflow/core:core_cpu",
#         "//tensorflow/core:core_cpu_internal",
#         "//tensorflow/core:framework",
#         "//tensorflow/core:lib",
#         "//tensorflow/core:protos_all_cc",
#         "//tensorflow/core:test",
#         "//tensorflow/core:test_main",
#         "//tensorflow/core:testlib",
#     ],
# )

tf_kernel_library(
    name = "reshape_op",
    prefix = "reshape_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "reverse_op",
    prefix = "reverse_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "reverse_sequence_op",
    prefix = "reverse_sequence_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "shape_ops",
    prefix = "shape_ops",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "slice_op",
    prefix = "slice_op",
    deps = ARRAY_DEPS + [":strided_slice_op"],
)

tf_kernel_library(
    name = "snapshot_op",
    prefix = "snapshot_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "split_op",
    gpu_srcs = ["cuda_device_array.h"],
    prefix = "split_op",
    deps = ARRAY_DEPS + [":split_lib"],
)

tf_kernel_library(
    name = "split_v_op",
    gpu_srcs = ["cuda_device_array.h"],
    prefix = "split_v_op",
    deps = ARRAY_DEPS + [":split_lib"],
)

tf_kernel_library(
    name = "searchsorted_op",
    prefix = "searchsorted_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "inplace_ops",
    prefix = "inplace_ops",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "tile_ops",
    srcs = ["tile_functor_cpu.cc"],
    hdrs = ["tile_functor.h"],
    gpu_srcs = [
        "tile_functor.h",
        "tile_functor_gpu.cu.cc",
    ],
    prefix = "tile_ops",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "transpose_op",
    srcs = [
        "transpose_op.cc",
    ],
    hdrs = ["transpose_op.h"],
    deps = ARRAY_DEPS + if_mkl([":mkl_transpose_op"]),
)

tf_kernel_library(
    name = "unique_op",
    prefix = "unique_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "unpack_op",
    prefix = "unpack_op",
    deps = ARRAY_DEPS + [":split_lib"],
)

tf_kernel_library(
    name = "unravel_index_op",
    prefix = "unravel_index_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "where_op",
    srcs = ["where_op.cc"],
    hdrs = ["where_op.h"],
    gpu_srcs = [
        "where_op.h",
        "where_op_gpu.cu.h",
        "where_op_gpu_impl_1.cu.cc",
        "where_op_gpu_impl_2.cu.cc",
        "where_op_gpu_impl_3.cu.cc",
        "where_op_gpu_impl_4.cu.cc",
        "where_op_gpu_impl_5.cu.cc",
    ],
    deps = if_cuda([
        ":cuda_solvers",
        "@cub_archive//:cub",
    ]) + ARRAY_DEPS,
)

cc_library(
    name = "ragged_ops",
    deps = [
        ":ragged_gather_op",
        ":ragged_range_op",
        ":ragged_tensor_to_sparse_kernel",
    ],
)

tf_kernel_library(
    name = "ragged_gather_op",
    srcs = ["ragged_gather_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:ragged_array_ops_op_lib",
    ],
)

tf_cc_test(
    name = "ragged_gather_op_test",
    size = "small",
    srcs = ["ragged_gather_op_test.cc"],
    deps = [
        ":ragged_gather_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:ragged_array_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_kernel_library(
    name = "ragged_range_op",
    srcs = ["ragged_range_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:ragged_math_ops_op_lib",
    ],
)

tf_cc_test(
    name = "ragged_range_op_test",
    srcs = ["ragged_range_op_test.cc"],
    deps = [
        ":ragged_range_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:ragged_math_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_kernel_library(
    name = "ragged_tensor_to_sparse_kernel",
    srcs = ["ragged_tensor_to_sparse_kernel.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:ragged_conversion_ops_op_lib",
    ],
)

tf_cc_test(
    name = "ragged_tensor_to_sparse_kernel_test",
    size = "small",
    srcs = ["ragged_tensor_to_sparse_kernel_test.cc"],
    deps = [
        ":ragged_tensor_to_sparse_kernel",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ragged_conversion_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_kernel_library(
    name = "cudnn_rnn_kernels",
    srcs = ["cudnn_rnn_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_util_hdrs",
        "//tensorflow/core:cudnn_rnn_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core/kernels:bounds_check_lib",
        "//third_party/eigen3",
        "@farmhash_archive//:farmhash",
    ],
)

tf_cc_test(
    name = "batch_norm_op_test",
    size = "small",
    srcs = ["batch_norm_op_test.cc"],
    deps = [
        ":batch_norm_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "ops_testutil_test",
    size = "small",
    srcs = ["ops_testutil_test.cc"],
    deps = [
        ":identity_op",
        ":ops_testutil",
        ":ops_util",
        ":variable_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "concat_op_test",
    size = "small",
    srcs = ["concat_op_test.cc"],
    deps = [
        ":concat_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "bincount_op_test",
    size = "small",
    srcs = ["bincount_op_test.cc"],
    deps = [
        ":bincount_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.cc"],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "deep_conv2d_test",
    size = "small",
    srcs = ["deep_conv2d_test.cc"],
    deps = [
        ":conv_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "xsmm_conv2d_test",
    size = "small",
    srcs = select({
        ":xsmm_convolutions": ["xsmm_conv2d_test.cc"],
        "//conditions:default": [],
    }),
    deps = [
        ":conv_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + select({
        ":xsmm_convolutions": [
            "@libxsmm_archive//:xsmm_avx",
        ],
        "//conditions:default": [],
    }),
)

tf_cc_test(
    name = "conv_ops_test",
    size = "medium",
    srcs = ["conv_ops_test.cc"],
    deps = [
        ":conv_ops",
        ":image",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "depthwise_conv_ops_test",
    size = "small",
    srcs = ["depthwise_conv_ops_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":conv_ops",
        ":image",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "decode_wav_op_test",
    size = "small",
    srcs = ["decode_wav_op_test.cc"],
    deps = [
        ":decode_wav_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "encode_wav_op_test",
    size = "small",
    srcs = ["encode_wav_op_test.cc"],
    deps = [
        ":decode_wav_op",
        ":encode_wav_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "example_parsing_ops_test",
    size = "medium",
    srcs = ["example_parsing_ops_test.cc"],
    shard_count = 4,
    tags = ["optonly"],
    deps = [
        ":example_parsing_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "fake_quant_ops_test",
    size = "small",
    srcs = ["fake_quant_ops_test.cc"],
    deps = [
        ":fake_quant_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "fused_batch_norm_op_test",
    size = "small",
    srcs = ["fused_batch_norm_op_test.cc"],
    deps = [
        ":fused_batch_norm_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "gather_functor",
    prefix = "gather_functor",
    visibility = [":friends"],
    deps = [
        ":bounds_check",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

# Unlike gather_functor library, this does not include the CUDA code and deps.
cc_library(
    name = "gather_functor_hdr",
    hdrs = ["gather_functor.h"],
)

tf_kernel_library(
    name = "dense_update_functor",
    srcs = ["dense_update_functor.cc"],
    hdrs = ["dense_update_functor.h"],
    gpu_srcs = [
        "dense_update_functor.h",
        "dense_update_functor_gpu.cu.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
    alwayslink = 0,
)

tf_cuda_cc_test(
    name = "gather_op_test",
    size = "small",
    srcs = ["gather_op_test.cc"],
    deps = [
        ":gather_op",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "gather_nd_op_test",
    size = "small",
    srcs = ["gather_nd_op_test.cc"],
    deps = [
        ":gather_nd_op",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "guarantee_const_op_test",
    size = "small",
    srcs = ["guarantee_const_op_test.cc"],
    deps = [
        ":guarantee_const_op",
        ":ops_testutil",
        ":ops_util",
        ":variable_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "identity_op_test",
    size = "small",
    srcs = ["identity_op_test.cc"],
    deps = [
        ":identity_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "identity_n_op_test",
    size = "small",
    srcs = ["identity_n_op_test.cc"],
    deps = [
        ":identity_n_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "debug_ops_test",
    size = "small",
    srcs = ["debug_ops_test.cc"],
    deps = [
        ":debug_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:debug_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/debug:debug_io_utils",
        "//tensorflow/core/debug:debug_node_key",
    ],
)

tf_cuda_cc_test(
    name = "quantize_and_dequantize_op_test",
    size = "small",
    srcs = ["quantize_and_dequantize_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantize_and_dequantize_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "dequantize_op_test",
    size = "small",
    srcs = ["dequantize_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "reverse_op_test",
    size = "small",
    srcs = ["reverse_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":reverse_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "scatter_functor",
    prefix = "scatter_functor",
    visibility = [":friends"],
    deps = [
        ":bounds_check",
        ":dense_update_functor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "slice_op_test",
    size = "small",
    srcs = ["slice_op_test.cc"],
    linkopts = select({
        "//tensorflow:darwin": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":slice_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "strided_slice_op_test",
    size = "small",
    srcs = ["strided_slice_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":slice_op",
        ":strided_slice_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "unique_op_test",
    size = "small",
    srcs = ["unique_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":unique_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "transpose_functor",
    srcs = ["transpose_functor_cpu.cc"],
    hdrs = ["transpose_functor.h"],
    gpu_srcs = [
        "transpose_functor_gpu.cu.cc",
        "transpose_functor.h",
    ],
    visibility = [":friends"],
    deps = [
        ":conv_2d",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "transpose_util_test",
    size = "small",
    srcs = ["transpose_util_test.cc"],
    deps = [
        ":transpose_functor",
        "//tensorflow/core:framework",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_kernel_library(
    name = "candidate_sampler_ops",
    prefix = "candidate_sampler_ops",
    deps = [
        ":range_sampler",
        "//tensorflow/core:candidate_sampling_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "range_sampler",
    srcs = ["range_sampler.cc"],
    hdrs = ["range_sampler.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "range_sampler_test",
    size = "small",
    srcs = ["range_sampler_test.cc"],
    deps = [
        ":range_sampler",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_kernel_library(
    name = "control_flow_ops",
    prefix = "control_flow_ops",
    deps = [
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "ctc_ops",
    prefix = "ctc",
    deps = [
        ":bounds_check",
        ":ops_util",
        "//tensorflow/core:ctc_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/ctc:ctc_beam_search_lib",
        "//tensorflow/core/util/ctc:ctc_loss_calculator_lib",
    ],
)

tf_cc_test(
    name = "control_flow_ops_test",
    size = "small",
    srcs = ["control_flow_ops_test.cc"],
    deps = [
        ":control_flow_ops",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "data_flow",
    deps = [
        ":barrier_ops",
        ":conditional_accumulator_base_op",
        ":conditional_accumulator_op",
        ":dynamic_partition_op",
        ":dynamic_stitch_op",
        ":fifo_queue_op",
        ":lookup_table_init_op",
        ":lookup_table_op",
        ":map_stage_op",
        ":padding_fifo_queue_op",
        ":priority_queue_op",
        ":queue_ops",
        ":random_shuffle_queue_op",
        ":record_input_op",
        ":session_ops",
        ":sparse_conditional_accumulator_op",
        ":stack_ops",
        ":stage_op",
        ":tensor_array_ops",
    ],
)

cc_library(
    name = "lookup",
    deps = [
        ":lookup_table_init_op",
        ":lookup_table_op",
    ],
)

cc_header_only_library(
    name = "lookup_headers_lib",
    deps = [":lookup"],
)

DATA_FLOW_DEPS = [
    ":bounds_check",
    ":concat_lib",
    ":conditional_accumulator",
    ":conditional_accumulator_base",
    ":fifo_queue",
    ":initializable_lookup_table",
    ":lookup_util",
    ":padding_fifo_queue",
    ":priority_queue",
    ":queue_base",
    ":queue_op",
    ":sparse_conditional_accumulator",
    ":split_lib",
    ":tensor_array",
    ":typed_conditional_accumulator_base",
    ":typed_queue",
    "//third_party/eigen3",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:data_flow_ops_op_lib",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
]

tf_kernel_library(
    name = "conditional_accumulator_base_op",
    prefix = "conditional_accumulator_base_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "conditional_accumulator_op",
    prefix = "conditional_accumulator_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "barrier_ops",
    prefix = "barrier_ops",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "fifo_queue_op",
    prefix = "fifo_queue_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "padding_fifo_queue_op",
    prefix = "padding_fifo_queue_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "priority_queue_op",
    prefix = "priority_queue_op",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "queue_ops",
    prefix = "queue_ops",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "random_shuffle_queue_op",
    prefix = "random_shuffle_queue_op",
    deps = DATA_FLOW_DEPS + [
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "scoped_allocator_ops",
    prefix = "scoped_allocator_ops",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:scoped_allocator_ops_op_lib",
    ],
)

tf_cuda_cc_test(
    name = "scoped_allocator_ops_test",
    srcs = ["scoped_allocator_ops_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),  #Required for benchmarking
    deps = [
        ":cwise_op",
        ":dense_update_ops",
        ":ops_testutil",
        ":ops_util",
        ":scoped_allocator_ops",
        ":variable_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "session_ops",
    prefix = "session_ops",
    deps = DATA_FLOW_DEPS,
)

tf_kernel_library(
    name = "sparse_conditional_accumulator_op",
    prefix = "sparse_conditional_accumulator_op",
    deps = DATA_FLOW_DEPS,
)

cc_library(
    name = "stack",
    srcs = ["stack.cc"],
    hdrs = ["stack.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "stack_ops",
    prefix = "stack_ops",
    deps = DATA_FLOW_DEPS + [":stack"],
)

tf_kernel_library(
    name = "tensor_array_ops",
    prefix = "tensor_array_ops",
    deps = DATA_FLOW_DEPS,
)

DYNAMIC_DEPS = [
    ":bounds_check",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:data_flow_ops_op_lib",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
]

tf_kernel_library(
    name = "dynamic_partition_op",
    prefix = "dynamic_partition_op",
    deps = DYNAMIC_DEPS + [
        ":fill_functor",
        ":gather_functor",
    ] + if_cuda(["@cub_archive//:cub"]),
)

tf_kernel_library(
    name = "dynamic_stitch_op",
    gpu_srcs = [
        "cuda_device_array.h",
        "cuda_device_array_gpu.h",
    ],
    prefix = "dynamic_stitch_op",
    deps = DYNAMIC_DEPS,
)

LOOKUP_DEPS = [
    ":bounds_check",
    ":initializable_lookup_table",
    ":lookup_util",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:lookup_ops_op_lib",
]

tf_kernel_library(
    name = "lookup_table_init_op",
    prefix = "lookup_table_init_op",
    deps = LOOKUP_DEPS,
)

tf_kernel_library(
    name = "lookup_table_op",
    prefix = "lookup_table_op",
    deps = LOOKUP_DEPS,
)

cc_library(
    name = "checkpoint_ops",
    deps = [
        ":generate_vocab_remapping_op",
        ":load_and_remap_matrix_op",
    ],
)

tf_kernel_library(
    name = "generate_vocab_remapping_op",
    srcs = ["generate_vocab_remapping_op.cc"],
    deps = [
        ":lookup_table_init_op",
        ":lookup_table_op",
        "//tensorflow/core:checkpoint_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "load_and_remap_matrix_op",
    srcs = ["load_and_remap_matrix_op.cc"],
    deps = [
        "//tensorflow/core:checkpoint_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/util/tensor_bundle",
        "//third_party/eigen3",
    ],
)

tf_cuda_cc_tests(
    name = "dynamic_op_test",
    size = "small",
    srcs = [
        "dynamic_partition_op_test.cc",
        "dynamic_stitch_op_test.cc",
    ],
    deps = [
        ":data_flow",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "eye_functor",
    hdrs = ["eye_functor.h"],
    gpu_srcs = [
        "eye_functor_gpu.cu.cc",
        "eye_functor.h",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
    alwayslink = 0,
)

cc_library(
    name = "fifo_queue",
    srcs = ["fifo_queue.cc"],
    hdrs = ["fifo_queue.h"],
    visibility = [":friends"],
    deps = [
        ":queue_base",
        ":queue_op",
        ":typed_queue",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "padding_fifo_queue",
    srcs = ["padding_fifo_queue.cc"],
    hdrs = ["padding_fifo_queue.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":fifo_queue",
        ":queue_base",
        ":typed_queue",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "conditional_accumulator_base",
    srcs = ["conditional_accumulator_base.cc"],
    hdrs = [
        "conditional_accumulator_base.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "typed_conditional_accumulator_base",
    hdrs = ["typed_conditional_accumulator_base.h"],
    deps = [
        ":conditional_accumulator_base",
    ],
)

cc_library(
    name = "conditional_accumulator",
    hdrs = [
        "conditional_accumulator.h",
        "conditional_accumulator_base_op.h",
    ],
    deps = [
        ":conditional_accumulator_base",
        ":fill_functor",
        ":typed_conditional_accumulator_base",
    ],
)

cc_library(
    name = "sparse_conditional_accumulator",
    hdrs = ["sparse_conditional_accumulator.h"],
    deps = [
        ":typed_conditional_accumulator_base",
    ],
)

tf_kernel_library(
    name = "tensor_array",
    srcs = ["tensor_array.cc"],
    hdrs = ["tensor_array.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":aggregate_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "resource_variable_ops",
    srcs = ["resource_variable_ops.cc"],
    hdrs = ["resource_variable_ops.h"],
    deps = [
        ":bounds_check",
        ":dense_update_functor",
        ":gather_functor",
        ":mutex_ops",
        ":scatter_functor",
        ":state",
        ":training_op_helpers",
        ":variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:resource_variable_ops_op_lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "list_kernels",
    srcs = ["list_kernels.cc"],
    hdrs = ["list_kernels.h"],
    gpu_srcs = [
        "list_kernels.cu.cc",
        "list_kernels.h",
    ],
    deps = [
        ":concat_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:list_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "fact_op",
    prefix = "fact_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:user_ops_op_lib",
    ],
)

tf_kernel_library(
    name = "function_ops",
    prefix = "function_ops",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "functional_ops",
    prefix = "functional_ops",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "partitioned_function_ops",
    prefix = "partitioned_function_ops",
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/grappler/utils:functions",
    ],
)

cc_library(
    name = "image",
    deps = [
        ":adjust_contrast_op",
        ":adjust_hue_op",
        ":adjust_saturation_op",
        ":attention_ops",
        ":colorspace_op",
        ":crop_and_resize_op",
        ":decode_bmp_op",
        ":decode_image_op",
        ":draw_bounding_box_op",
        ":encode_jpeg_op",
        ":encode_png_op",
        ":extract_jpeg_shape_op",
        ":non_max_suppression_op",
        ":random_crop_op",
        ":resize_area_op",
        ":resize_bicubic_op",
        ":resize_bilinear_op",
        ":resize_nearest_neighbor_op",
        ":sample_distorted_bounding_box_op",
    ],
)

IMAGE_DEPS = [
    ":bounds_check",
    ":eigen_helpers",
    ":image_resizer_state",
    "//third_party/eigen3",
    "//tensorflow/core:framework",
    "//tensorflow/core:gif_internal",
    "//tensorflow/core:image_ops_op_lib",
    "//tensorflow/core:jpeg_internal",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:png_internal",
    "//tensorflow/core:protos_all_cc",
]

tf_kernel_library(
    name = "adjust_contrast_op",
    prefix = "adjust_contrast_op",
    deps = IMAGE_DEPS,
)

cc_library(
    name = "adjust_hsv_gpu_lib",
    hdrs = ["adjust_hsv_gpu.cu.h"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "adjust_hue_op",
    prefix = "adjust_hue_op",
    deps = IMAGE_DEPS + [":adjust_hsv_gpu_lib"],
)

tf_kernel_library(
    name = "adjust_saturation_op",
    prefix = "adjust_saturation_op",
    deps = IMAGE_DEPS + [":adjust_hsv_gpu_lib"],
)

tf_kernel_library(
    name = "attention_ops",
    prefix = "attention_ops",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "colorspace_op",
    prefix = "colorspace_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "crop_and_resize_op",
    prefix = "crop_and_resize_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "decode_bmp_op",
    prefix = "decode_bmp_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "decode_image_op",
    prefix = "decode_image_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "draw_bounding_box_op",
    prefix = "draw_bounding_box_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "encode_jpeg_op",
    prefix = "encode_jpeg_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "encode_png_op",
    prefix = "encode_png_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "extract_jpeg_shape_op",
    prefix = "extract_jpeg_shape_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "non_max_suppression_op",
    prefix = "non_max_suppression_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "random_crop_op",
    prefix = "random_crop_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "resize_area_op",
    prefix = "resize_area_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "resize_bicubic_op",
    prefix = "resize_bicubic_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "resize_bilinear_op",
    prefix = "resize_bilinear_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "resize_nearest_neighbor_op",
    prefix = "resize_nearest_neighbor_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "sample_distorted_bounding_box_op",
    prefix = "sample_distorted_bounding_box_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "encode_wav_op",
    prefix = "encode_wav_op",
    deps = [
        ":bounds_check",
        "//tensorflow/core:audio_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "decode_wav_op",
    prefix = "decode_wav_op",
    deps = [
        "//tensorflow/core:audio_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_tests(
    name = "eigen_test",
    size = "small",
    srcs = [
        "eigen_activations_test.cc",
        "eigen_attention_test.cc",
        "eigen_backward_spatial_convolutions_test.cc",
        "eigen_pooling_test.cc",
        "eigen_softmax_test.cc",
        "eigen_spatial_convolutions_test.cc",
    ],
    deps = [
        ":eigen_helpers",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Conditional test target generation is not supported by the "tf_cc_tests" macro
# (can't add 'select' to the srcs field, type 'select' is not iterable).
tf_cc_test(
    name = "eigen_mkldnn_contraction_kernel_test",
    size = "small",
    srcs = select({
        ":mkldnn_contraction_kernel": ["eigen_mkldnn_contraction_kernel_test.cc"],
        "//conditions:default": [],
    }),
    tags = ["mkldnn_contraction_kernel"],
    deps = [
        ":eigen_contraction_kernel",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "eigen_benchmark",
    testonly = 1,
    hdrs = [
        "eigen_benchmark.h",
        ":eigen_helpers",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "eigen_benchmark_cpu_test",
    srcs = ["eigen_benchmark_cpu_test.cc"],
    deps = [
        ":eigen_benchmark",
        ":eigen_helpers",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

tf_cc_tests(
    name = "basic_ops_benchmark_test",
    size = "small",
    srcs = [
        "basic_ops_benchmark_test.cc",
    ],
    deps = [
        ":math",
        ":ops_util",
        ":state",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_tests(
    name = "bonus_tests",
    srcs = [
        "adjust_contrast_op_test.cc",
        "colorspace_op_test.cc",
        "crop_and_resize_op_test.cc",
        "non_max_suppression_op_test.cc",
        "resize_area_op_test.cc",
        "resize_bicubic_op_test.cc",
        "resize_bilinear_op_test.cc",
        "resize_nearest_neighbor_op_test.cc",
    ],
    linkopts = select({
        "//tensorflow:darwin": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        ":image",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "adjust_contrast_op_benchmark_test",
    srcs = ["adjust_contrast_op_benchmark_test.cc"],
    deps = [
        ":image",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "crop_and_resize_op_benchmark_test",
    srcs = ["crop_and_resize_op_benchmark_test.cc"],
    deps = [
        ":image",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "resize_benchmark_test",
    srcs = ["resize_op_benchmark_test.cc"],
    deps = [
        ":image",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "io",
    deps = [
        ":fixed_length_record_reader_op",
        ":identity_reader_op",
        ":lmdb_reader_op",
        ":matching_files_op",
        ":reader_ops",
        ":restore_op",
        ":save_op",
        ":save_restore_v2_ops",
        ":text_line_reader_op",
        ":tf_record_reader_op",
        ":whole_file_read_ops",
    ],
)

IO_DEPS = [
    ":ops_util",
    "//tensorflow/core:framework",
    "//tensorflow/core:io_ops_op_lib",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core:reader_base",
    "//tensorflow/core/util/tensor_bundle",
]

tf_kernel_library(
    name = "fixed_length_record_reader_op",
    prefix = "fixed_length_record_reader_op",
    deps = IO_DEPS,
)

tf_kernel_library(
    name = "identity_reader_op",
    prefix = "identity_reader_op",
    deps = IO_DEPS,
)

tf_kernel_library(
    name = "lmdb_reader_op",
    prefix = "lmdb_reader_op",
    deps = IO_DEPS + [
        "@lmdb",
    ],
)

tf_kernel_library(
    name = "matching_files_op",
    prefix = "matching_files_op",
    deps = IO_DEPS,
)

tf_kernel_library(
    name = "reader_ops",
    prefix = "reader_ops",
    deps = IO_DEPS,
)

SAVE_RESTORE_DEPS = [
    ":bounds_check_lib",
    ":save_restore_tensor",
    "//tensorflow/core:framework",
    "//tensorflow/core:io_ops_op_lib",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/util/tensor_bundle",
]

tf_kernel_library(
    name = "restore_op",
    prefix = "restore_op",
    deps = SAVE_RESTORE_DEPS,
)

tf_kernel_library(
    name = "save_op",
    prefix = "save_op",
    deps = SAVE_RESTORE_DEPS,
)

tf_kernel_library(
    name = "save_restore_v2_ops",
    prefix = "save_restore_v2_ops",
    deps = SAVE_RESTORE_DEPS,
)

tf_kernel_library(
    name = "text_line_reader_op",
    prefix = "text_line_reader_op",
    deps = IO_DEPS,
)

tf_kernel_library(
    name = "tf_record_reader_op",
    prefix = "tf_record_reader_op",
    deps = IO_DEPS,
)

tf_kernel_library(
    name = "whole_file_read_ops",
    prefix = "whole_file_read_ops",
    deps = IO_DEPS,
)

tf_cc_tests(
    name = "bonus2_tests",
    size = "small",
    srcs = [
        "merge_v2_checkpoints_op_test.cc",
        "restore_op_test.cc",
        "restore_v2_op_test.cc",
        "save_op_test.cc",
        "save_v2_op_test.cc",
    ],
    deps = [
        ":io",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

cc_library(
    name = "linalg",
    deps = [
        ":cholesky_grad",
        ":cholesky_op",
        ":determinant_op",
        ":matrix_exponential_op",
        ":matrix_inverse_op",
        ":matrix_logarithm_op",
        ":matrix_solve_ls_op",
        ":matrix_solve_op",
        ":matrix_square_root_op",
        ":matrix_triangular_solve_op",
        ":qr_op",
        ":self_adjoint_eig_op",
        ":self_adjoint_eig_v2_op",
        ":svd_op",
    ],
)

tf_kernel_library(
    name = "cuda_solvers",
    srcs = ["cuda_solvers.cc"],
    hdrs = ["cuda_solvers.h"],
    # @local_config_cuda//cuda:cusolver, //third_party/eigen3:blas,
    # and //third_party/libf2c all contain various parts of BLAS, LAPACK,
    # and f2c helper functions in global namespace. Tell the compiler to
    # allow multiple definitions when linking this.
    linkopts = select({
        "//tensorflow:darwin": [],
        "//tensorflow:windows": [],
        "//conditions:default": ["-Wl,-z,muldefs"],
    }),
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform/default/build_config:cublas_plugin",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cusolver",
    ],
)

LINALG_DEPS = [
    ":linalg_ops_common",
    "//third_party/eigen3",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:linalg_ops_op_lib",
] + if_cuda([
    ":cuda_solvers",
    ":transpose_functor",
])

tf_kernel_library(
    name = "cholesky_op",
    prefix = "cholesky_op",
    deps = if_cuda([
        ":matrix_band_part_op",
    ]) + LINALG_DEPS,
)

tf_kernel_library(
    name = "cholesky_grad",
    prefix = "cholesky_grad",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "determinant_op",
    prefix = "determinant_op",
    deps = if_cuda([
        ":fill_functor",
    ]) + LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_exponential_op",
    prefix = "matrix_exponential_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_logarithm_op",
    prefix = "matrix_logarithm_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "self_adjoint_eig_op",
    prefix = "self_adjoint_eig_op",
    deps = LINALG_DEPS + ["//tensorflow/core:lib_internal"],
)

tf_kernel_library(
    name = "self_adjoint_eig_v2_op",
    prefix = "self_adjoint_eig_v2_op",
    deps = LINALG_DEPS + ["//tensorflow/core:lib_internal"] + if_cuda([
        ":cast_op",
        ":cwise_op",
    ]),
)

tf_kernel_library(
    name = "matrix_inverse_op",
    prefix = "matrix_inverse_op",
    deps = LINALG_DEPS + if_cuda([":eye_functor"]),
)

tf_kernel_library(
    name = "matrix_solve_ls_op",
    prefix = "matrix_solve_ls_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_solve_op",
    prefix = "matrix_solve_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_square_root_op",
    prefix = "matrix_square_root_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_triangular_solve_op",
    prefix = "matrix_triangular_solve_op",
    deps = LINALG_DEPS + if_cuda([
        "//tensorflow/core/platform/default/build_config:cublas_plugin",
    ]),
)

tf_kernel_library(
    name = "qr_op",
    prefix = "qr_op",
    deps = LINALG_DEPS + if_cuda([
        ":cwise_op",
        ":eye_functor",
        ":matrix_band_part_op",
    ]),
)

tf_kernel_library(
    name = "svd_op",
    prefix = "svd_op",
    deps = LINALG_DEPS,
)

cc_library(
    name = "linalg_ops_common",
    srcs = ["linalg_ops_common.cc"],
    hdrs = ["linalg_ops_common.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "logging",
    deps = [
        ":logging_ops",
        ":summary_audio_op",
        ":summary_image_op",
        ":summary_op",
        ":summary_tensor_op",
    ],
)

LOGGING_DEPS = [
    "@com_google_absl//absl/strings",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:logging_ops_op_lib",
    "//tensorflow/core:protos_all_cc",
]

tf_kernel_library(
    name = "logging_ops",
    prefix = "logging_ops",
    deps = LOGGING_DEPS,
)

tf_kernel_library(
    name = "summary_audio_op",
    prefix = "summary_audio_op",
    deps = LOGGING_DEPS,
)

tf_kernel_library(
    name = "summary_image_op",
    prefix = "summary_image_op",
    deps = LOGGING_DEPS + ["//tensorflow/core:png_internal"],
)

tf_kernel_library(
    name = "summary_op",
    prefix = "summary_op",
    deps = LOGGING_DEPS,
)

tf_kernel_library(
    name = "summary_tensor_op",
    prefix = "summary_tensor_op",
    deps = LOGGING_DEPS,
)

tf_cc_tests(
    name = "bonus3_tests",
    size = "small",
    srcs = [
        "logging_ops_test.cc",
        "summary_audio_op_test.cc",
        "summary_image_op_test.cc",
        "summary_op_test.cc",
        "summary_tensor_op_test.cc",
    ],
    deps = [
        ":logging",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "manip",
    deps = [
        ":roll_op",
    ],
)

tf_kernel_library(
    name = "roll_op",
    prefix = "roll_op",
    deps = [
        ":bounds_check",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:manip_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "roll_op_test",
    size = "small",
    srcs = ["roll_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":roll_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

MATH_DEPS = [
    ":bounds_check",
    ":fill_functor",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:math_grad",
    "//tensorflow/core:math_ops_op_lib",
    "//third_party/eigen3",
]

cc_library(
    name = "math_not_windows",
    deps = [
        ":sparse_matmul_op",
    ],
)

tf_kernel_library(
    name = "sparse_matmul_op",
    defines = select({
        ":xsmm": ["TENSORFLOW_USE_LIBXSMM"],
        "//conditions:default": [],
    }),
    prefix = "sparse_matmul_op",
    deps = MATH_DEPS + select({
        ":xsmm": [
            "@libxsmm_archive//:xsmm_avx",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "math",
    deps = [
        ":aggregate_ops",
        ":argmax_op",
        ":batch_matmul_op",
        ":betainc_op",
        ":bincount_op",
        ":bucketize_op",
        ":cast_op",
        ":check_numerics_op",
        ":compare_and_bitpack_op",
        ":cross_op",
        ":cwise_op",
        ":fft_ops",
        ":histogram_op",
        ":matmul_op",
        ":population_count_op",
        ":reduction_ops",
        ":scan_ops",
        ":segment_reduction_ops",
        ":sequence_ops",
    ],
)

tf_kernel_library(
    name = "aggregate_ops",
    prefix = "aggregate_ops",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "argmax_op",
    prefix = "argmax_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "batch_matmul_op",
    srcs = if_mkl_ml([
        "mkl_batch_matmul_op.cc",
    ]),
    # <prefix>*impl.h are excluded by default from the CPU build, add explicitly.
    hdrs = ["batch_matmul_op_impl.h"],
    prefix = "batch_matmul_op",
    deps = MATH_DEPS + [":eigen_contraction_kernel"] + if_mkl_ml([
        "//third_party/mkl:intel_binary_blob",
    ]),
)

tf_kernel_library(
    name = "betainc_op",
    prefix = "betainc_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "bucketize_op",
    gpu_srcs = ["cuda_device_array.h"],
    prefix = "bucketize_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "cast_op",
    prefix = "cast_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "check_numerics_op",
    prefix = "check_numerics_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "cross_op",
    prefix = "cross_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "cwise_op",
    prefix = "cwise_op",
    deps = MATH_DEPS + ["//tensorflow/core:bitwise_ops_op_lib"],
)

tf_kernel_library(
    name = "population_count_op",
    prefix = "population_count_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "fft_ops",
    prefix = "fft_ops",
    deps = MATH_DEPS + [
        "//tensorflow/core:spectral_ops_op_lib",
    ] + if_cuda([
        "//tensorflow/core/platform/default/build_config:cufft_plugin",
    ]),
)

tf_kernel_library(
    name = "matmul_op",
    srcs = [
        "matmul_op.cc",
    ] + if_mkl([
        "mkl_matmul_op.cc",
    ]),
    hdrs = ["matmul_op.h"],
    defines = select({
        ":xsmm": [
            "TENSORFLOW_USE_LIBXSMM",
            "EIGEN_USE_LIBXSMM",
        ],
        "//conditions:default": [],
    }),
    deps = MATH_DEPS + [
        ":eigen_contraction_kernel",
        ":gpu_util_hdrs",
    ] + select({
        ":xsmm": ["@libxsmm_archive//:xsmm_avx"],
        "//conditions:default": [],
    }) + mkl_deps() + if_cuda([
        "//tensorflow/core/platform/default/build_config:cublas_plugin",
    ]),
)

tf_kernel_library(
    name = "reduction_ops",
    gpu_srcs = ["reduction_gpu_kernels.cu.h"],
    prefix = "reduction_ops",
    deps = MATH_DEPS + [":transpose_functor"] + if_cuda(["@cub_archive//:cub"]),
)

tf_kernel_library(
    name = "segment_reduction_ops",
    prefix = "segment_reduction_ops",
    deps = MATH_DEPS + if_cuda([
        ":cuda_solvers",
    ]),
)

tf_kernel_library(
    name = "scan_ops",
    prefix = "scan_ops",
    deps = MATH_DEPS + if_cuda(["@cub_archive//:cub"]),
)

tf_kernel_library(
    name = "sequence_ops",
    prefix = "sequence_ops",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "unary_ops_composition",
    prefix = "unary_ops_composition",
    deps = MATH_DEPS + [
        ":cwise_op",
        ":relu_op",
    ],
)

tf_cc_test(
    name = "sequence_ops_test",
    size = "small",
    srcs = ["sequence_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":sequence_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "cast_op_test",
    size = "small",
    srcs = ["cast_op_test.cc"],
    deps = [
        ":cast_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "cross_op_test",
    size = "small",
    srcs = ["cross_op_test.cc"],
    deps = [
        ":cross_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_tests(
    name = "sparse_tests",
    size = "small",
    srcs = [
        "sparse_add_op_test.cc",
        "sparse_dense_binary_op_shared_test.cc",
        "sparse_reduce_sum_op_test.cc",
    ],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":sparse_add_op",
        ":sparse_dense_binary_op_shared",
        ":sparse_reduce_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "cwise_ops_test",
    size = "small",
    srcs = ["cwise_ops_test.cc"],
    deps = [
        ":bounds_check",
        ":cwise_op",
        ":nn",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "unary_ops_composition_test",
    size = "small",
    srcs = ["unary_ops_composition_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":unary_ops_composition",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "matmul_op_test",
    size = "small",
    srcs = ["matmul_op_test.cc"],
    deps = [
        ":matmul_op",
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "batch_matmul_op_test",
    size = "small",
    srcs = ["batch_matmul_op_test.cc"],
    deps = [
        ":batch_matmul_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "scan_ops_test",
    size = "small",
    srcs = ["scan_ops_test.cc"],
    linkopts = select({
        "//tensorflow:darwin": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        ":scan_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "reduction_ops_test",
    size = "small",
    srcs = ["reduction_ops_test.cc"],
    linkopts = select({
        "//tensorflow:darwin": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        ":reduction_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "segment_reduction_ops_test",
    size = "small",
    srcs = ["segment_reduction_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":segment_reduction_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "immutable_constant_op_test",
    srcs = ["immutable_constant_op_test.cc"],
    deps = [
        ":array",
        ":immutable_constant_op",
        ":matmul_op",
        ":ops_testutil",
        ":ops_util",
        ":random_shuffle_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "shape_op_test",
    srcs = ["shape_op_test.cc"],
    deps = [
        ":array",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "sparse_matmul_op_test",
    size = "small",
    srcs = ["sparse_matmul_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":sparse_matmul_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "split_op_test",
    size = "small",
    srcs = ["split_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":split_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "split_v_op_test",
    size = "small",
    srcs = ["split_v_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":split_v_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "diag_op_test",
    size = "small",
    srcs = ["diag_op_test.cc"],
    deps = [
        ":diag_op",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# conv_grad_ops currently has to be built with conv_ops*.
# TODO(josh11b, zhengxq): put these a separate libraries in ":nn" below once
# conv_ops_gpu.h has be separated into its own library.
tf_kernel_library(
    name = "conv_ops",
    srcs = [
        "conv_grad_filter_ops.cc",
        "conv_grad_input_ops.cc",
        "conv_grad_ops.cc",
        "conv_grad_ops_3d.cc",
        "deep_conv2d.cc",
    ] + select({
        ":xsmm_convolutions": ["xsmm_conv2d.cc"],
        "//conditions:default": [],
    }),
    hdrs = [
        "fill_functor.h",
        "conv_grad_ops.h",
        "deep_conv2d.h",
        "gemm_functors.h",
        "winograd_transform.h",
    ] + select({
        ":xsmm_convolutions": ["xsmm_conv2d.h"],
        "//conditions:default": [],
    }),
    defines = select({
        ":xsmm_convolutions": [
            "TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS",
        ],
        "//conditions:default": [],
    }) + select({
        ":xsmm": ["EIGEN_USE_LIBXSMM"],
        "//conditions:default": [],
    }) + select({
        ":xsmm_backward_convolutions": ["TENSORFLOW_USE_LIBXSMM_BACKWARD_CONVOLUTIONS"],
        "//conditions:default": [],
    }),
    prefix = "conv_ops",
    deps = [
        ":bounds_check",
        ":conv_2d",
        ":conv_3d",
        ":eigen_contraction_kernel",
        ":image_resizer_state",
        ":fill_functor",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_ops_op_lib",
    ] + select({
        ":xsmm_convolutions": [
            "@libxsmm_archive//:xsmm_avx",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "//tensorflow/core/platform/default/build_config:cublas_plugin",
        "//tensorflow/core/platform/default/build_config:cudnn_plugin",
    ]),
)

tf_kernel_library(
    name = "depthwise_conv_op",
    prefix = "depthwise_conv_op",
    deps = [
        ":bounds_check",
        ":conv_ops",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
    ] + if_cuda([
        "@cub_archive//:cub",
        "@local_config_cuda//cuda:cudnn_header",
    ]),
)

tf_kernel_library(
    name = "depthwise_conv_grad_op",
    hdrs = [
        "depthwise_conv_op.h",
    ],
    prefix = "depthwise_conv_grad_op",
    deps = [
        ":bounds_check",
        ":conv_ops",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
    ] + if_cuda([
        "@local_config_cuda//cuda:cudnn_header",
    ]),
)

cc_library(
    name = "nn",
    deps = [
        ":batch_norm_op",
        ":bias_op",
        ":conv_ops",
        ":data_format_ops",
        ":depthwise_conv_grad_op",
        ":depthwise_conv_op",
        ":dilation_ops",
        ":fused_batch_norm_op",
        ":in_topk_op",
        ":l2loss_op",
        ":lrn_op",
        ":nth_element_op",
        ":relu_op",
        ":softmax_op",
        ":softplus_op",
        ":softsign_op",
        ":topk_op",
        ":xent_op",
    ],
)

# Kernels for the nodes intented to be added to the graph by the Grappler optimizers.
cc_library(
    name = "grappler",
    deps = [
        ":unary_ops_composition",
    ],
)

NN_DEPS = [
    ":bounds_check",
    ":conv_2d",
    ":eigen_contraction_kernel",
    ":fused_batch_norm_util_gpu",
    ":ops_util",
    ":pooling_ops",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:nn_grad",
    "//tensorflow/core:nn_ops_op_lib",
    "//third_party/eigen3",
] + if_mkl(["//tensorflow/core:mkl_nn_ops_op_lib"])

tf_kernel_library(
    name = "batch_norm_op",
    prefix = "batch_norm_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "data_format_ops",
    prefix = "data_format_ops",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "bias_op",
    prefix = "bias_op",
    deps = NN_DEPS + if_cuda([
        ":reduction_ops",
        "@cub_archive//:cub",
    ]),
)

tf_kernel_library(
    name = "fused_batch_norm_op",
    prefix = "fused_batch_norm_op",
    deps = NN_DEPS + [
        ":fill_functor",
    ],
)

tf_kernel_library(
    name = "in_topk_op",
    prefix = "in_topk_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "lrn_op",
    prefix = "lrn_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "relu_op",
    prefix = "relu_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "softmax_op",
    prefix = "softmax_op",
    deps = NN_DEPS + if_cuda([
        ":reduction_ops",
        "@cub_archive//:cub",
    ]),
)

tf_kernel_library(
    name = "softplus_op",
    prefix = "softplus_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "softsign_op",
    prefix = "softsign_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "topk_op",
    prefix = "topk_op",
    deps = NN_DEPS + if_cuda(["@cub_archive//:cub"]),
)

tf_kernel_library(
    name = "nth_element_op",
    prefix = "nth_element_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "xent_op",
    prefix = "xent_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "bincount_op",
    prefix = "bincount_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/eigen3",
    ] + if_cuda(["@cub_archive//:cub"]),
)

tf_kernel_library(
    name = "histogram_op",
    prefix = "histogram_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/eigen3",
    ] + if_cuda(["@cub_archive//:cub"]),
)

tf_kernel_library(
    name = "l2loss_op",
    prefix = "l2loss_op",
    deps = [
        ":reduction_ops",
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_grad",
        "//tensorflow/core:nn_ops_op_lib",
    ] + if_cuda(["@cub_archive//:cub"]),
)

tf_cuda_cc_test(
    name = "lrn_op_test",
    srcs = ["lrn_op_test.cc"],
    deps = [
        ":nn",
        ":ops_testutil",
        ":ops_util",
        ":xent_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "xent_op_test",
    srcs = ["xent_op_test.cc"],
    deps = [
        ":nn",
        ":ops_testutil",
        ":ops_util",
        ":xent_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "nn_ops_test",
    srcs = ["nn_ops_test.cc"],
    deps = [
        ":host_constant_op",
        ":nn",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "pooling_ops",
    srcs = [
        "avgpooling_op.cc",
        "cudnn_pooling_gpu.cc",
        "fractional_avg_pool_op.cc",
        "fractional_max_pool_op.cc",
        "fractional_pool_common.cc",
        "maxpooling_op.cc",
        "pooling_ops_3d.cc",
        "pooling_ops_common.cc",
    ],
    hdrs = [
        "avgpooling_op.h",
        "cudnn_pooling_gpu.h",
        "fractional_pool_common.h",
        "maxpooling_op.h",
        "pooling_ops_3d.h",
        "pooling_ops_common.h",
    ] + if_sycl(["pooling_ops_3d_sycl.h"]),
    gpu_srcs = [
        "avgpooling_op.h",
        "avgpooling_op_gpu.cu.cc",
        "maxpooling_op.h",
        "maxpooling_op_gpu.cu.cc",
        "maxpooling_op_gpu.h",
        "pooling_ops_common.h",
        "pooling_ops_common_gpu.h",
        "pooling_ops_3d_gpu.h",
        "pooling_ops_3d_gpu.cu.cc",
    ],
    deps = [
        ":bounds_check",
        ":conv_2d",
        ":conv_3d",
        ":conv_ops",
        ":eigen_helpers",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:stream_executor",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "fake_quant_ops",
    srcs = ["fake_quant_ops.cc"],
    hdrs = ["fake_quant_ops_functor.h"],
    gpu_srcs = [
        "fake_quant_ops_gpu.cu.cc",
        "fake_quant_ops_functor.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "fused_batch_norm_util",
    gpu_srcs = [
        "fused_batch_norm_op.h",
        "fused_batch_norm_op.cu.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "pooling_ops_hdrs",
    hdrs = [
        "avgpooling_op.h",
        "maxpooling_op.h",
        "pooling_ops_common.h",
    ],
    deps = [
        ":eigen_helpers",
        ":ops_util_hdrs",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "dilation_ops",
    prefix = "dilation_ops",
    deps = [
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "batch_space_ops",
    srcs = [
        "batchtospace_op.cc",
        "spacetobatch_functor.cc",
        "spacetobatch_functor.h",
        "spacetobatch_op.cc",
    ],
    gpu_srcs = [
        "spacetobatch_functor.h",
        "spacetobatch_functor_gpu.cu.cc",
    ],
    visibility = [":friends"],
    deps = [
        ":bounds_check",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_cuda_cc_test(
    name = "spacetobatch_benchmark_test",
    srcs = ["spacetobatch_benchmark_test.cc"],
    deps = [
        ":batch_space_ops",
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "depth_space_ops",
    srcs = [
        "depthtospace_op.cc",
        "spacetodepth_op.cc",
    ],
    hdrs = [
        "depthtospace_op.h",
        "spacetodepth_op.h",
    ],
    gpu_srcs = [
        "depthtospace_op.h",
        "depthtospace_op_gpu.cu.cc",
        "spacetodepth_op.h",
        "spacetodepth_op_gpu.cu.cc",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "parsing",
    deps = [
        ":decode_compressed_op",
        ":decode_csv_op",
        ":decode_raw_op",
        ":example_parsing_ops",
        ":parse_tensor_op",
        ":string_to_number_op",
    ],
)

PARSING_DEPS = [
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:parsing_ops_op_lib",
    "//tensorflow/core:proto_text",
    "//tensorflow/core:protos_all_cc",
]

tf_kernel_library(
    name = "decode_csv_op",
    prefix = "decode_csv_op",
    deps = PARSING_DEPS,
)

tf_kernel_library(
    name = "decode_raw_op",
    prefix = "decode_raw_op",
    deps = PARSING_DEPS,
)

tf_kernel_library(
    name = "decode_compressed_op",
    prefix = "decode_compressed_op",
    deps = [
        "//tensorflow/core:lib_internal",
    ] + PARSING_DEPS,
)

tf_kernel_library(
    name = "example_parsing_ops",
    prefix = "example_parsing_ops",
    deps = PARSING_DEPS,
)

tf_kernel_library(
    name = "parse_tensor_op",
    prefix = "parse_tensor_op",
    deps = PARSING_DEPS,
)

tf_cc_test(
    name = "parse_tensor_test",
    srcs = ["parse_tensor_test.cc"],
    deps = [
        ":ops_testutil",
        ":parse_tensor_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "string_to_number_op",
    prefix = "string_to_number_op",
    deps = PARSING_DEPS,
)

cc_library(
    name = "random_ops",
    deps = [
        ":random_op",
        ":random_shuffle_op",
    ],
)

RANDOM_OPS_DEPS = [
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:random_ops_op_lib",
]

tf_kernel_library(
    name = "random_op",
    prefix = "random_op",
    deps = RANDOM_OPS_DEPS,
)

tf_kernel_library(
    name = "random_shuffle_op",
    prefix = "random_shuffle_op",
    deps = RANDOM_OPS_DEPS,
)

tf_cuda_cc_test(
    name = "random_op_test",
    size = "small",
    srcs = ["random_op_test.cc"],
    deps = [
        ":host_constant_op",
        ":random_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "stateless_random_ops",
    prefix = "stateless_random_ops",
    deps = [
        ":bounds_check",
        ":random_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:stateless_random_ops_op_lib",
    ],
)

cc_library(
    name = "required",
    deps = [
        ":no_op",
        ":sendrecv_ops",
    ],
)

REQUIRED_DEPS = [
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:no_op_op_lib",
    "//tensorflow/core:sendrecv_ops_op_lib",
]

tf_kernel_library(
    name = "no_op",
    prefix = "no_op",
    deps = REQUIRED_DEPS,
)

tf_kernel_library(
    name = "sendrecv_ops",
    prefix = "sendrecv_ops",
    deps = REQUIRED_DEPS,
)

tf_cc_test(
    name = "sendrecv_ops_test",
    srcs = ["sendrecv_ops_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),  # Required for benchmarking
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":sendrecv_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "sparse",
    deps = [
        ":deserialize_sparse_string_op",
        ":deserialize_sparse_variant_op",
        ":serialize_sparse_op",
        ":sparse_add_grad_op",
        ":sparse_add_op",
        ":sparse_concat_op",
        ":sparse_cross_op",
        ":sparse_dense_binary_op_shared",
        ":sparse_fill_empty_rows_op",
        ":sparse_reduce_op",
        ":sparse_reorder_op",
        ":sparse_reshape_op",
        ":sparse_slice_grad_op",
        ":sparse_slice_op",
        ":sparse_softmax",
        ":sparse_sparse_binary_op_shared",
        ":sparse_split_op",
        ":sparse_tensor_dense_add_op",
        ":sparse_tensor_dense_matmul_op",
        ":sparse_tensors_map_ops",
        ":sparse_to_dense_op",
        ":sparse_xent_op",
    ],
)

SPARSE_DEPS = [
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:sparse_ops_op_lib",
]

tf_kernel_library(
    name = "sparse_add_grad_op",
    prefix = "sparse_add_grad_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_add_op",
    prefix = "sparse_add_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_concat_op",
    prefix = "sparse_concat_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_fill_empty_rows_op",
    prefix = "sparse_fill_empty_rows_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_cross_op",
    prefix = "sparse_cross_op",
    deps = SPARSE_DEPS + [
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_reduce_op",
    prefix = "sparse_reduce_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_dense_binary_op_shared",
    prefix = "sparse_dense_binary_op_shared",
    deps = SPARSE_DEPS + [
        ":cwise_op",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_sparse_binary_op_shared",
    prefix = "sparse_sparse_binary_op_shared",
    deps = SPARSE_DEPS + [
        ":cwise_op",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_reorder_op",
    prefix = "sparse_reorder_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_reshape_op",
    prefix = "sparse_reshape_op",
    deps = SPARSE_DEPS + [
        ":reshape_util",
    ],
)

tf_kernel_library(
    name = "sparse_slice_grad_op",
    prefix = "sparse_slice_grad_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_slice_op",
    prefix = "sparse_slice_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_softmax",
    prefix = "sparse_softmax",
    deps = SPARSE_DEPS + [
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_split_op",
    prefix = "sparse_split_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_tensor_dense_add_op",
    prefix = "sparse_tensor_dense_add_op",
    deps = SPARSE_DEPS + [
        ":scatter_functor",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_tensor_dense_matmul_op",
    prefix = "sparse_tensor_dense_matmul_op",
    deps = SPARSE_DEPS + [
        ":bounds_check",
        ":fill_functor",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_to_dense_op",
    prefix = "sparse_to_dense_op",
    deps = SPARSE_DEPS + [
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "sparse_xent_op",
    prefix = "sparse_xent_op",
    deps = SPARSE_DEPS + [
        ":bounds_check",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "serialize_sparse_op",
    prefix = "serialize_sparse_op",
    deps = SPARSE_DEPS + [
        ":reshape_util",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "deserialize_sparse_string_op",
    prefix = "deserialize_sparse_string_op",
    deps = SPARSE_DEPS + [
        ":reshape_util",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "deserialize_sparse_variant_op",
    prefix = "deserialize_sparse_variant_op",
    deps = SPARSE_DEPS + [
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "sparse_tensors_map_ops",
    prefix = "sparse_tensors_map_ops",
    deps = SPARSE_DEPS,
)

tf_cuda_cc_tests(
    name = "sparse2_tests",
    size = "small",
    srcs = [
        "sparse_tensor_dense_matmul_op_test.cc",
        "sparse_to_dense_op_test.cc",
        "sparse_xent_op_test.cc",
    ],
    deps = [
        ":host_constant_op",
        ":ops_testutil",
        ":ops_util",
        ":sparse",
        ":xent_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "loss_updaters",
    hdrs = [
        "hinge-loss.h",
        "logistic-loss.h",
        "loss.h",
        "poisson-loss.h",
        "smooth-hinge-loss.h",
        "squared-loss.h",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "loss_test",
    size = "small",
    srcs = ["loss_test.cc"],
    deps = [
        ":loss_updaters",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "sdca_ops_test",
    size = "small",
    srcs = ["sdca_ops_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),  # Required for benchmarking
    deps = [
        ":ops_util",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "sdca_ops",
    prefix = "sdca_ops",
    deps = [
        ":loss_updaters",
        ":sdca_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:sdca_ops_op_lib",
        "//third_party/eigen3",
        "@farmhash_archive//:farmhash",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sdca_internal",
    srcs = ["sdca_internal.cc"],
    hdrs = ["sdca_internal.h"],
    deps = [
        ":loss_updaters",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "state",
    deps = [
        ":count_up_to_op",
        ":dense_update_ops",
        ":scatter_nd_op",
        ":scatter_op",
        ":variable_ops",
    ],
)

STATE_DEPS = [
    ":assign_op",
    ":bounds_check",
    ":fill_functor",
    ":scatter_functor",
    "//third_party/eigen3",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:state_ops_op_lib",
] + if_sycl(["//tensorflow/core:sycl_runtime"])

tf_kernel_library(
    name = "count_up_to_op",
    prefix = "count_up_to_op",
    deps = STATE_DEPS + [":variable_ops"],
)

tf_kernel_library(
    name = "dense_update_ops",
    prefix = "dense_update_ops",
    deps = STATE_DEPS + [":dense_update_functor"],
)

tf_kernel_library(
    name = "scatter_op",
    prefix = "scatter_op",
    deps = STATE_DEPS,
)

tf_kernel_library(
    name = "scatter_nd_op",
    srcs = [
        "scatter_nd_op.cc",
        "scatter_nd_op_cpu_impl_0.cc",
        "scatter_nd_op_cpu_impl_1.cc",
        "scatter_nd_op_cpu_impl_2.cc",
        "scatter_nd_op_cpu_impl_3.cc",
        "scatter_nd_op_cpu_impl_4.cc",
        "scatter_nd_op_cpu_impl_5.cc",
        "scatter_nd_op_cpu_impl_6.cc",
        "scatter_nd_op_cpu_impl_7.cc",
    ],
    hdrs = [
        "scatter_nd_op.h",
        "scatter_nd_op_cpu_impl.h",
    ],
    gpu_srcs = [
        "scatter_nd_op.h",
        "scatter_nd_op_gpu.cu.cc",
    ],
    deps = STATE_DEPS + [
        ":dense_update_functor",
        ":training_op_helpers",
        ":variable_ops",
    ],
)

tf_kernel_library(
    name = "variable_ops",
    prefix = "variable_ops",
    deps = STATE_DEPS,
)

tf_kernel_library(
    name = "mutex_ops",
    prefix = "mutex_ops",
    deps = STATE_DEPS + [":ops_util"],
)

tf_cc_test(
    name = "scatter_op_test",
    size = "small",
    srcs = ["scatter_op_test.cc"],
    deps = [
        ":fill_functor",
        ":ops_testutil",
        ":ops_util",
        ":scatter_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cuda_cc_test(
    name = "scatter_nd_op_test",
    size = "small",
    srcs = ["scatter_nd_op_test.cc"],
    tags = ["noasan"],  # http://b/32635055
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":scatter_nd_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "string",
    deps = [
        ":as_string_op",
        ":base64_ops",
        ":reduce_join_op",
        ":regex_full_match_op",
        ":regex_replace_op",
        ":string_format_op",
        ":string_join_op",
        ":string_length_op",
        ":string_split_op",
        ":string_strip_op",
        ":string_to_hash_bucket_op",
        ":substr_op",
        ":unicode_ops",
        ":unicode_script_op",
    ],
)

cc_library(
    name = "string_util",
    srcs = ["string_util.cc"],
    hdrs = ["string_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@icu//:common",
    ],
)

STRING_DEPS = [
    ":bounds_check",
    ":string_util",
    "//third_party/eigen3",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:string_ops_op_lib",
]

tf_kernel_library(
    name = "string_to_hash_bucket_op",
    prefix = "string_to_hash_bucket_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "reduce_join_op",
    prefix = "reduce_join_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "string_format_op",
    prefix = "string_format_op",
    deps = STRING_DEPS + ["@com_google_absl//absl/strings"],
)

tf_cc_test(
    name = "string_format_op_test",
    size = "small",
    srcs = ["string_format_op_test.cc"],
    deps = [
        ":string_format_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_kernel_library(
    name = "string_join_op",
    prefix = "string_join_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "string_length_op",
    prefix = "string_length_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "regex_full_match_op",
    prefix = "regex_full_match_op",
    deps = STRING_DEPS + ["@com_googlesource_code_re2//:re2"],
)

tf_kernel_library(
    name = "regex_replace_op",
    prefix = "regex_replace_op",
    deps = STRING_DEPS + ["@com_googlesource_code_re2//:re2"],
)

tf_cc_test(
    name = "regex_replace_op_test",
    size = "small",
    srcs = ["regex_replace_op_test.cc"],
    deps = [
        ":regex_replace_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_kernel_library(
    name = "string_split_op",
    prefix = "string_split_op",
    deps = STRING_DEPS,
)

tf_cc_test(
    name = "string_split_op_test",
    size = "small",
    srcs = ["string_split_op_test.cc"],
    deps = [
        ":string_split_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_kernel_library(
    name = "string_strip_op",
    prefix = "string_strip_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "substr_op",
    prefix = "substr_op",
    deps = STRING_DEPS,
)

tf_cc_test(
    name = "substr_op_test",
    size = "small",
    srcs = ["substr_op_test.cc"],
    deps = [
        ":substr_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_kernel_library(
    name = "as_string_op",
    prefix = "as_string_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "unicode_ops",
    prefix = "unicode_ops",
    deps = [
        ":string_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:string_ops_op_lib",
        "//third_party/icu/data:conversion_data",
        "@icu//:common",
    ],
)

tf_kernel_library(
    name = "base64_ops",
    prefix = "base64_ops",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "training_ops",
    prefix = "training_ops",
    deps = [
        ":bounds_check",
        ":training_op_helpers",
        ":variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:training_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "training_ops_test",
    size = "small",
    srcs = ["training_ops_test.cc"],
    deps = [
        ":dense_update_ops",
        ":ops_util",
        ":training_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "multinomial_op",
    prefix = "multinomial_op",
    deps = [
        ":random_op",
        ":random_ops",
        ":stateless_random_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/eigen3",
    ],
)

tf_cuda_cc_test(
    name = "multinomial_op_test",
    size = "small",
    srcs = ["multinomial_op_test.cc"],
    deps = [
        ":host_constant_op",
        ":multinomial_op",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "parameterized_truncated_normal_op",
    prefix = "parameterized_truncated_normal_op",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:random_ops_op_lib",
    ],
)

tf_cuda_cc_test(
    name = "parameterized_truncated_normal_op_test",
    size = "small",
    srcs = ["parameterized_truncated_normal_op_test.cc"],
    deps = [
        ":host_constant_op",
        ":ops_util",
        ":parameterized_truncated_normal_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "random_poisson_op",
    prefix = "random_poisson_op",
    deps = [
        ":random_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:random_ops_op_lib",
    ],
)

tf_cuda_cc_test(
    name = "random_poisson_op_test",
    size = "small",
    srcs = ["random_poisson_op_test.cc"],
    deps = [
        ":ops_util",
        ":random_poisson_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "word2vec_kernels",
    prefix = "word2vec_kernels",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:word2vec_ops",
    ],
)

filegroup(
    name = "spectrogram_test_data",
    srcs = [
        "spectrogram_test_data/short_test_segment.wav",
        "spectrogram_test_data/short_test_segment_spectrogram.csv.bin",
        "spectrogram_test_data/short_test_segment_spectrogram_400_200.csv.bin",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "spectrogram",
    srcs = ["spectrogram.cc"],
    hdrs = ["spectrogram.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/fft2d:fft2d_headers",
        "@fft2d",
    ],
)

cc_library(
    name = "spectrogram_test_utils",
    testonly = 1,
    srcs = ["spectrogram_test_utils.cc"],
    hdrs = ["spectrogram_test_utils.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
    ],
)

tf_cc_binary(
    name = "spectrogram_convert_test_data",
    testonly = 1,
    srcs = ["spectrogram_convert_test_data.cc"],
    deps = [
        ":spectrogram_test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "spectrogram_test",
    size = "medium",
    srcs = ["spectrogram_test.cc"],
    data = [":spectrogram_test_data"],
    deps = [
        ":spectrogram",
        ":spectrogram_test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "spectrogram_op",
    prefix = "spectrogram_op",
    deps = [
        ":spectrogram",
        "//tensorflow/core:audio_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "spectrogram_op_test",
    size = "small",
    srcs = ["spectrogram_op_test.cc"],
    deps = [
        ":ops_util",
        ":spectrogram_op",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "mfcc_dct",
    srcs = ["mfcc_dct.cc"],
    hdrs = ["mfcc_dct.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "mfcc_dct_test",
    size = "small",
    srcs = ["mfcc_dct_test.cc"],
    deps = [
        ":mfcc_dct",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "mfcc_mel_filterbank",
    srcs = ["mfcc_mel_filterbank.cc"],
    hdrs = ["mfcc_mel_filterbank.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "mfcc_mel_filterbank_test",
    size = "small",
    srcs = ["mfcc_mel_filterbank_test.cc"],
    deps = [
        ":mfcc_mel_filterbank",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "mfcc",
    srcs = ["mfcc.cc"],
    hdrs = ["mfcc.h"],
    copts = tf_copts(),
    deps = [
        ":mfcc_dct",
        ":mfcc_mel_filterbank",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "mfcc_test",
    size = "small",
    srcs = ["mfcc_test.cc"],
    deps = [
        ":mfcc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "mfcc_op",
    prefix = "mfcc_op",
    deps = [
        ":mfcc",
        "//tensorflow/core:audio_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "mfcc_op_test",
    size = "small",
    srcs = ["mfcc_op_test.cc"],
    deps = [
        ":mfcc_op",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "audio",
    deps = [
        ":decode_wav_op",
        ":encode_wav_op",
        ":mfcc_op",
        ":spectrogram_op",
    ],
)

# Android libraries -----------------------------------------------------------

# Changes to the Android srcs here should be replicated in
# tensorflow/contrib/makefile/tf_op_files.txt
# LINT.IfChange
filegroup(
    name = "mobile_srcs",
    srcs = [
        "avgpooling_op.h",
        "batch_util.h",
        "bounds_check.h",
        "cwise_ops.h",
        "cwise_ops_common.h",
        "cwise_ops_gradients.h",
        "eigen_activations.h",
        "eigen_attention.h",
        "eigen_backward_cuboid_convolutions.h",
        "eigen_backward_spatial_convolutions.h",
        "eigen_cuboid_convolution.h",
        "eigen_pooling.h",
        "eigen_softmax.h",
        "eigen_spatial_convolutions.h",
        "eigen_volume_patch.h",
        "fifo_queue.h",
        "maxpooling_op.h",
        "ops_util.h",
        "padding_fifo_queue.h",
        "pooling_ops_common.cc",
        "pooling_ops_common.h",
        "queue_base.h",
        "queue_op.h",
        "typed_queue.h",
    ],
)

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
)

# Core kernels we want on Android. Only a subset of kernels to keep
# base library small.
filegroup(
    name = "android_core_ops",
    srcs = [
        "aggregate_ops.cc",
        "aggregate_ops.h",
        "aggregate_ops_cpu.h",
        "assign_op.h",
        "bias_op.cc",
        "bias_op.h",
        "bounds_check.h",
        "cast_op.cc",
        "cast_op.h",
        "cast_op_impl.h",
        "cast_op_impl_bfloat.cc",
        "cast_op_impl_bool.cc",
        "cast_op_impl_complex128.cc",
        "cast_op_impl_complex64.cc",
        "cast_op_impl_double.cc",
        "cast_op_impl_float.cc",
        "cast_op_impl_half.cc",
        "cast_op_impl_int16.cc",
        "cast_op_impl_int32.cc",
        "cast_op_impl_int64.cc",
        "cast_op_impl_int8.cc",
        "cast_op_impl_uint16.cc",
        "cast_op_impl_uint32.cc",
        "cast_op_impl_uint64.cc",
        "cast_op_impl_uint8.cc",
        "concat_lib.h",
        "concat_lib_cpu.cc",
        "concat_lib_cpu.h",
        "concat_op.cc",
        "constant_op.cc",
        "constant_op.h",
        "cwise_ops.h",
        "cwise_ops_common.cc",
        "cwise_ops_common.h",
        "cwise_ops_gradients.h",
        "dense_update_functor.cc",
        "dense_update_functor.h",
        "dense_update_ops.cc",
        "example_parsing_ops.cc",
        "fill_functor.cc",
        "fill_functor.h",
        "function_ops.cc",
        "function_ops.h",
        "gather_functor.h",
        "gather_nd_op.cc",
        "gather_nd_op.h",
        "gather_nd_op_cpu_impl.h",
        "gather_nd_op_cpu_impl_0.cc",
        "gather_nd_op_cpu_impl_1.cc",
        "gather_nd_op_cpu_impl_2.cc",
        "gather_nd_op_cpu_impl_3.cc",
        "gather_nd_op_cpu_impl_4.cc",
        "gather_nd_op_cpu_impl_5.cc",
        "gather_nd_op_cpu_impl_6.cc",
        "gather_nd_op_cpu_impl_7.cc",
        "gather_op.cc",
        "identity_n_op.cc",
        "identity_n_op.h",
        "identity_op.cc",
        "identity_op.h",
        "immutable_constant_op.cc",
        "immutable_constant_op.h",
        "matmul_op.cc",
        "matmul_op.h",
        "no_op.cc",
        "no_op.h",
        "non_max_suppression_op.cc",
        "non_max_suppression_op.h",
        "one_hot_op.cc",
        "one_hot_op.h",
        "ops_util.h",
        "pack_op.cc",
        "pooling_ops_common.h",
        "reshape_op.cc",
        "reshape_op.h",
        "reverse_sequence_op.cc",
        "reverse_sequence_op.h",
        "sendrecv_ops.cc",
        "sendrecv_ops.h",
        "sequence_ops.cc",
        "shape_ops.cc",
        "shape_ops.h",
        "slice_op.cc",
        "slice_op.h",
        "slice_op_cpu_impl.h",
        "slice_op_cpu_impl_1.cc",
        "slice_op_cpu_impl_2.cc",
        "slice_op_cpu_impl_3.cc",
        "slice_op_cpu_impl_4.cc",
        "slice_op_cpu_impl_5.cc",
        "slice_op_cpu_impl_6.cc",
        "slice_op_cpu_impl_7.cc",
        "softmax_op.cc",
        "softmax_op_functor.h",
        "split_lib.h",
        "split_lib_cpu.cc",
        "split_op.cc",
        "split_v_op.cc",
        "strided_slice_op.cc",
        "strided_slice_op.h",
        "strided_slice_op_impl.h",
        "strided_slice_op_inst_0.cc",
        "strided_slice_op_inst_1.cc",
        "strided_slice_op_inst_2.cc",
        "strided_slice_op_inst_3.cc",
        "strided_slice_op_inst_4.cc",
        "strided_slice_op_inst_5.cc",
        "strided_slice_op_inst_6.cc",
        "strided_slice_op_inst_7.cc",
        "unpack_op.cc",
        "variable_ops.cc",
        "variable_ops.h",
    ],
)

# Other kernels we may want on Android.
#
# The kernels can be consumed as a whole or in two groups for
# supporting separate compilation. Note that the split into groups
# is entirely for improving compilation time, and not for
# organizational reasons; you should not depend on any
# of those groups independently.
filegroup(
    name = "android_extended_ops",
    srcs = [
        ":android_extended_ops_group1",
        ":android_extended_ops_group2",
        ":android_quantized_ops",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "android_extended_ops_headers",
    srcs = [
        "argmax_op.h",
        "avgpooling_op.h",
        "batch_matmul_op_impl.h",
        "batch_norm_op.h",
        "control_flow_ops.h",
        "conv_2d.h",
        "conv_3d.h",
        "conv_ops.h",
        "conv_ops_gpu.h",
        "data_format_ops.h",
        "depthtospace_op.h",
        "depthwise_conv_op.h",
        "fake_quant_ops_functor.h",
        "fused_batch_norm_op.h",
        "gemm_functors.h",
        "image_resizer_state.h",
        "initializable_lookup_table.h",
        "lookup_table_init_op.h",
        "lookup_table_op.h",
        "lookup_util.h",
        "maxpooling_op.h",
        "mfcc.h",
        "mfcc_dct.h",
        "mfcc_mel_filterbank.h",
        "mirror_pad_op.h",
        "mirror_pad_op_cpu_impl.h",
        "multinomial_op.h",
        "pad_op.h",
        "pooling_ops_3d.h",
        "random_op.h",
        "reduction_ops.h",
        "reduction_ops_common.h",
        "relu_op.h",
        "relu_op_functor.h",
        "reshape_util.h",
        "resize_bilinear_op.h",
        "resize_nearest_neighbor_op.h",
        "reverse_op.h",
        "save_restore_tensor.h",
        "segment_reduction_ops.h",
        "softplus_op.h",
        "softsign_op.h",
        "spacetobatch_functor.h",
        "spacetodepth_op.h",
        "spectrogram.h",
        "stateless_random_ops.h",
        "string_util.h",
        "tensor_array.h",
        "tile_functor.h",
        "tile_ops_cpu_impl.h",
        "tile_ops_impl.h",
        "topk_op.h",
        "training_op_helpers.h",
        "training_ops.h",
        "transpose_functor.h",
        "transpose_op.h",
        "where_op.h",
        "xent_op.h",
    ],
)

filegroup(
    name = "android_extended_ops_group1",
    srcs = [
        "argmax_op.cc",
        "avgpooling_op.cc",
        "batch_matmul_op_real.cc",
        "batch_norm_op.cc",
        "bcast_ops.cc",
        "check_numerics_op.cc",
        "control_flow_ops.cc",
        "conv_2d.h",
        "conv_grad_filter_ops.cc",
        "conv_grad_input_ops.cc",
        "conv_grad_ops.cc",
        "conv_grad_ops.h",
        "conv_ops.cc",
        "conv_ops_3d.cc",
        "conv_ops_fused.cc",
        "conv_ops_using_gemm.cc",
        "crop_and_resize_op.cc",
        "crop_and_resize_op.h",
        "cwise_op_abs.cc",
        "cwise_op_add_1.cc",
        "cwise_op_add_2.cc",
        "cwise_op_bitwise_and.cc",
        "cwise_op_bitwise_or.cc",
        "cwise_op_bitwise_xor.cc",
        "cwise_op_cos.cc",
        "cwise_op_cosh.cc",
        "cwise_op_div.cc",
        "cwise_op_equal_to_1.cc",
        "cwise_op_equal_to_2.cc",
        "cwise_op_not_equal_to_1.cc",
        "cwise_op_not_equal_to_2.cc",
        "cwise_op_exp.cc",
        "cwise_op_floor.cc",
        "cwise_op_floor_div.cc",
        "cwise_op_floor_mod.cc",
        "cwise_op_greater.cc",
        "cwise_op_greater_equal.cc",
        "cwise_op_invert.cc",
        "cwise_op_isfinite.cc",
        "cwise_op_isnan.cc",
        "cwise_op_left_shift.cc",
        "cwise_op_less.cc",
        "cwise_op_less_equal.cc",
        "cwise_op_log.cc",
        "cwise_op_logical_and.cc",
        "cwise_op_logical_not.cc",
        "cwise_op_logical_or.cc",
        "cwise_op_maximum.cc",
        "cwise_op_minimum.cc",
        "cwise_op_mul_1.cc",
        "cwise_op_mul_2.cc",
        "cwise_op_neg.cc",
        "cwise_op_pow.cc",
        "cwise_op_reciprocal.cc",
        "cwise_op_right_shift.cc",
        "cwise_op_round.cc",
        "cwise_op_rsqrt.cc",
        "cwise_op_select.cc",
        "cwise_op_sigmoid.cc",
        "cwise_op_sign.cc",
        "cwise_op_sin.cc",
        "cwise_op_sinh.cc",
        "cwise_op_sqrt.cc",
        "cwise_op_square.cc",
        "cwise_op_squared_difference.cc",
        "cwise_op_sub.cc",
        "cwise_op_tan.cc",
        "cwise_op_tanh.cc",
        "cwise_op_xlogy.cc",
        "cwise_op_xdivy.cc",
        "data_format_ops.cc",
        "decode_wav_op.cc",
        "deep_conv2d.cc",
        "deep_conv2d.h",
        "depthwise_conv_op.cc",
        "dynamic_partition_op.cc",
        "encode_wav_op.cc",
        "fake_quant_ops.cc",
        "fifo_queue.cc",
        "fifo_queue_op.cc",
        "fused_batch_norm_op.cc",
        "listdiff_op.cc",
        "population_count_op.cc",
        "population_count_op.h",
        "winograd_transform.h",
        ":android_extended_ops_headers",
    ] + select({
        ":xsmm_convolutions": [
            "xsmm_conv2d.h",
            "xsmm_conv2d.cc",
        ],
        "//conditions:default": [],
    }),
)

filegroup(
    name = "android_extended_ops_group2",
    srcs = [
        "batchtospace_op.cc",
        "ctc_decoder_ops.cc",
        "decode_bmp_op.cc",
        "depthtospace_op.cc",
        "dynamic_stitch_op.cc",
        "in_topk_op.cc",
        "initializable_lookup_table.cc",
        "logging_ops.cc",
        "lookup_table_init_op.cc",
        "lookup_table_op.cc",
        "lookup_util.cc",
        "lrn_op.cc",
        "maxpooling_op.cc",
        "mfcc.cc",
        "mfcc_dct.cc",
        "mfcc_mel_filterbank.cc",
        "mfcc_op.cc",
        "mirror_pad_op.cc",
        "mirror_pad_op_cpu_impl_1.cc",
        "mirror_pad_op_cpu_impl_2.cc",
        "mirror_pad_op_cpu_impl_3.cc",
        "mirror_pad_op_cpu_impl_4.cc",
        "mirror_pad_op_cpu_impl_5.cc",
        "multinomial_op.cc",
        "pad_op.cc",
        "padding_fifo_queue.cc",
        "padding_fifo_queue_op.cc",
        "pooling_ops_3d.cc",
        "queue_base.cc",
        "queue_op.cc",
        "queue_ops.cc",
        "random_op.cc",
        "reduction_ops_all.cc",
        "reduction_ops_any.cc",
        "reduction_ops_common.cc",
        "reduction_ops_max.cc",
        "reduction_ops_mean.cc",
        "reduction_ops_min.cc",
        "reduction_ops_prod.cc",
        "reduction_ops_sum.cc",
        "relu_op.cc",
        "reshape_util.cc",
        "resize_bilinear_op.cc",
        "resize_nearest_neighbor_op.cc",
        "restore_op.cc",
        "reverse_op.cc",
        "save_op.cc",
        "save_restore_tensor.cc",
        "save_restore_v2_ops.cc",
        "segment_reduction_ops.cc",
        "session_ops.cc",
        "softplus_op.cc",
        "softsign_op.cc",
        "spacetobatch_functor.cc",
        "spacetobatch_op.cc",
        "spacetodepth_op.cc",
        "sparse_fill_empty_rows_op.cc",
        "sparse_reshape_op.cc",
        "sparse_to_dense_op.cc",
        "spectrogram.cc",
        "spectrogram_op.cc",
        "stack.cc",
        "stack.h",
        "stack_ops.cc",
        "stateless_random_ops.cc",
        "string_join_op.cc",
        "string_util.cc",
        "summary_op.cc",
        "tensor_array.cc",
        "tensor_array_ops.cc",
        "tile_functor_cpu.cc",
        "tile_ops.cc",
        "tile_ops_cpu_impl_1.cc",
        "tile_ops_cpu_impl_2.cc",
        "tile_ops_cpu_impl_3.cc",
        "tile_ops_cpu_impl_4.cc",
        "tile_ops_cpu_impl_5.cc",
        "tile_ops_cpu_impl_6.cc",
        "tile_ops_cpu_impl_7.cc",
        "topk_op.cc",
        "training_op_helpers.cc",
        "training_ops.cc",
        "transpose_functor_cpu.cc",
        "transpose_op.cc",
        "unique_op.cc",
        "where_op.cc",
        "xent_op.cc",
        ":android_extended_ops_headers",
    ],
)

filegroup(
    name = "android_quantized_ops",
    srcs = [
        "dequantize_op.cc",
        "meta_support.cc",
        "meta_support.h",
        "quantization_utils.cc",
        "quantization_utils.h",
        "quantize_down_and_shrink_range.cc",
        "quantize_op.cc",
        "quantized_activation_ops.cc",
        "quantized_add_op.cc",
        "quantized_batch_norm_op.cc",
        "quantized_bias_add_op.cc",
        "quantized_concat_op.cc",
        "quantized_conv_ops.cc",
        "quantized_instance_norm.cc",
        "quantized_matmul_op.cc",
        "quantized_mul_op.cc",
        "quantized_pooling_ops.cc",
        "quantized_reshape_op.cc",
        "quantized_resize_bilinear_op.cc",
        "reference_gemm.h",
        "requantization_range_op.cc",
        "requantize.cc",
        "reshape_op.h",
    ],
    visibility = ["//visibility:public"],
)

ANDROID_TEXTUAL_HDRS = [
    "gather_nd_op_cpu_impl.h",
    "gemm_functors.h",
    "mirror_pad_op_cpu_impl.h",
    "scatter_nd_op_cpu_impl.h",
    "slice_op_cpu_impl.h",
    "strided_slice_op_impl.h",
    "tile_ops_cpu_impl.h",
]

# A file group which contains nearly all available operators which
# may work on Android. This is intended to be used with selective
# registration.
filegroup(
    name = "android_all_ops",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*test.cc",
            "*test_util*",
            "*testutil*",
            "*testlib*",
            "*main.cc",
            "*_gpu*",
            "*_3d*",
            "*.cu.*",
            # Ops already in android_srcs
            "pooling_ops_common.cc",
            # Ops which we are currently excluding because they are likely
            # not used on Android. Those ops also do not compile if included,
            # unless we add the additional deps they need.
            "tf_record_reader_op.*",
            "cudnn_rnn_ops.*",
            "lmdb_reader_op.*",
            "string_to_hash_bucket_op.*",
            "sdca_ops.*",
            "sdca_internal.*",
            "sparse_cross_op.*",
            "text_line_reader_op.*",
            "summary_image_op.*",
            "decode_image_op.*",
            "encode_png_op.*",
            "encode_jpeg_op.*",
            "extract_jpeg_shape_op.*",
            "decode_jpeg_op.*",
            "decode_and_crop_jpeg_op.*",
            "decode_gif_op.*",
            "identity_reader_op.*",
            "remote_fused_graph_execute_op.*",
            "remote_fused_graph_rewriter_transform.*",
            "fixed_length_record_reader_op.*",
            "whole_file_read_ops.*",
            "sample_distorted_bounding_box_op.*",
            "ctc_loss_op.*",
            "summary_interface.*",
            "summary_kernels.*",
            "spectrogram_convert_test_data.cc",
            "decode_proto_op.cc",
            "encode_proto_op.cc",
            "rpc_op.cc",
            "partitioned_function_ops.cc",
            # Excluded due to experimental status:
            "debug_ops.*",
            "mutex_ops.*",
            "batch_kernels.*",
            "regex_full_match_op.cc",
            "regex_replace_op.cc",
            "unicode_script_op.cc",
            "unicode_ops.cc",
            # Ops that are inherently incompatible with Android (e.g. tied to x86 platform).
            "mkl_*",
            "xsmm_*",
            "cwise_ops_sycl_common.h",
        ] + ANDROID_TEXTUAL_HDRS,
    ),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "android_all_ops_textual_hdrs",
    srcs = ANDROID_TEXTUAL_HDRS,
    visibility = ["//visibility:public"],
)
# LINT.ThenChange(//tensorflow/contrib/makefile/tf_op_files.txt)

cc_library(
    name = "android_tensorflow_kernels",
    srcs = select({
        "//tensorflow:android": [
            "//tensorflow/core/kernels:android_core_ops",
            "//tensorflow/core/kernels:android_extended_ops",
        ],
        "//conditions:default": [],
    }),
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:android": [
            "-ldl",
        ],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:android_tensorflow_lib_lite",
        "//tensorflow/core:protos_all_cc_impl",
        "//third_party/eigen3",
        "//third_party/fft2d:fft2d_headers",
        "@fft2d",
        "@gemmlowp",
        "@protobuf_archive//:protobuf",
    ],
    alwayslink = 1,
)

cc_library(
    name = "android_tensorflow_image_op",
    srcs = if_android(["decode_image_op.cc"]),
    copts = tf_copts(),
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:android_gif_internal",
        "//tensorflow/core:android_jpeg_internal",
        "//tensorflow/core:android_png_internal",
        "//tensorflow/core:android_tensorflow_lib_lite",
    ],
    alwayslink = 1,
)

cc_library(
    name = "android_whole_file_read_ops",
    srcs = if_android(["whole_file_read_ops.cc"]),
    copts = tf_copts(),
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:android_tensorflow_lib_lite",
    ],
    alwayslink = 1,
)

#   Quantization-specific OpKernels

tf_kernel_library(
    name = "quantized_ops",
    srcs = [
        "dequantize_op.cc",
        "meta_support.cc",
        "quantize_down_and_shrink_range.cc",
        "quantize_op.cc",
        "quantized_activation_ops.cc",
        "quantized_add_op.cc",
        "quantized_batch_norm_op.cc",
        "quantized_bias_add_op.cc",
        "quantized_concat_op.cc",
        "quantized_conv_ops.cc",
        "quantized_instance_norm.cc",
        "quantized_matmul_op.cc",
        "quantized_mul_op.cc",
        "quantized_pooling_ops.cc",
        "quantized_reshape_op.cc",
        "quantized_resize_bilinear_op.cc",
        "requantization_range_op.cc",
        "requantize.cc",
        "reshape_op.h",
    ],
    hdrs = [
        "meta_support.h",
        "reference_gemm.h",
    ],
    deps = [
        ":concat_lib_hdrs",
        ":conv_ops",
        ":cwise_op",
        ":eigen_helpers",
        ":image_resizer_state",
        ":ops_util",
        ":pooling_ops",
        ":quantization_utils",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//third_party/eigen3",
        "@gemmlowp",
    ],
)

tf_cc_test(
    name = "requantization_range_op_test",
    size = "small",
    srcs = ["requantization_range_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantize_down_and_shrink_range_op_test",
    size = "small",
    srcs = ["quantize_down_and_shrink_range_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "requantize_op_test",
    size = "small",
    srcs = ["requantize_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantization_utils_test",
    srcs = ["quantization_utils_test.cc"],
    deps = [
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
    ],
)

# Android-only test for quantization utilities.
tf_cc_binary(
    name = "quantization_utils_test_android_only",
    testonly = 1,
    srcs = ["quantization_utils_test.cc"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:android": [
            "-lm",
            "-llog",
            "-pie",
            "-std=c++11",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
    ] + select({
        "//tensorflow:android": [
            ":android_tensorflow_kernels",
            "//tensorflow/core:android_tensorflow_lib",
            "//tensorflow/core:android_tensorflow_test_lib",
        ],
        "//conditions:default": [
            ":quantized_ops",
            "//third_party/eigen3",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:test",
            "//tensorflow/cc:cc_ops",
            "//tensorflow/cc:client_session",
            "//tensorflow/core:framework",
            "//tensorflow/core:tensor_testutil",
        ],
    }),
)

tf_cc_test(
    name = "quantized_activation_ops_test",
    srcs = ["quantized_activation_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Android-only test for quantized addition.
cc_binary(
    name = "quantized_add_op_test_android_only",
    testonly = 1,
    srcs = ["quantized_add_op_test.cc"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:android": [
            "-lm",
            "-llog",
            "-pie",
            "-std=c++11",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
    ] + select({
        "//tensorflow:android": [
            ":android_tensorflow_kernels",
            "//tensorflow/core:android_tensorflow_lib",
            "//tensorflow/core:android_tensorflow_test_lib",
        ],
        "//conditions:default": [
            ":ops_util",
            ":quantized_ops",
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:tensor_testutil",
            "//tensorflow/core:tensorflow",
            "//tensorflow/core:test",
        ],
    }),
)

tf_cc_test(
    name = "quantized_add_op_test",
    size = "small",
    srcs = ["quantized_add_op_test.cc"],
    deps = [
        ":math",
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_resize_bilinear_op_test",
    size = "small",
    srcs = ["quantized_resize_bilinear_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:image_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

# Android-only test for quantized resize bilinear.
cc_binary(
    name = "quantized_resize_bilinear_op_test_android_only",
    testonly = 1,
    srcs = ["quantized_resize_bilinear_op_test.cc"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:android": [
            "-lm",
            "-llog",
            "-pie",
            "-std=c++11",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
    ] + select({
        "//tensorflow:android": [
            ":android_tensorflow_kernels",
            "//tensorflow/core:android_tensorflow_lib",
            "//tensorflow/core:android_tensorflow_test_lib",
        ],
        "//conditions:default": [
            ":ops_testutil",
            ":ops_util",
            ":quantized_ops",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:direct_session",
            "//tensorflow/core:framework",
            "//tensorflow/core:image_ops_op_lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:test",
            "//tensorflow/core:testlib",
        ],
    }),
)

tf_cc_test(
    name = "quantized_bias_add_op_test",
    size = "small",
    srcs = ["quantized_bias_add_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["quantized_conv_ops_test.cc"],
    tags = ["nomsan"],  # http://b/32242946
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantize_op_test",
    size = "small",
    srcs = ["quantize_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_matmul_op_test",
    size = "small",
    srcs = ["quantized_matmul_op_test.cc"],
    tags = ["nomsan"],  # http://b/32242946
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Android-only test for quantized multiply.
cc_binary(
    name = "quantized_mul_op_test_android_only",
    testonly = 1,
    srcs = ["quantized_mul_op_test.cc"],
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
    ] + select({
        "//tensorflow:android": [
            ":android_tensorflow_kernels",
            "//tensorflow/core:android_tensorflow_lib",
            "//tensorflow/core:android_tensorflow_test_lib",
        ],
        "//conditions:default": [
            ":ops_util",
            ":quantized_ops",
            "//tensorflow/core:framework",
            "//tensorflow/core:tensor_testutil",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:test",
        ],
    }),
)

tf_cc_test(
    name = "quantized_mul_op_test",
    size = "small",
    srcs = ["quantized_mul_op_test.cc"],
    deps = [
        ":math",
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_pooling_ops_test",
    size = "small",
    srcs = ["quantized_pooling_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test_mkl(
    name = "mkl_quantized_pooling_ops_test",
    size = "small",
    srcs = ["mkl_quantized_pooling_ops_test.cc"],
    deps = [
        ":mkl_input_conversion_op",
        ":mkl_pooling_ops",
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_reshape_op_test",
    size = "small",
    srcs = ["quantized_reshape_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_concat_op_test",
    size = "small",
    srcs = ["quantized_concat_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "quantized_batch_norm_op_test",
    size = "small",
    srcs = ["quantized_batch_norm_op_test.cc"],
    deps = [
        ":batch_norm_op",
        ":ops_testutil",
        ":quantization_utils",
        ":quantized_ops",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
    ],
)

# Android-only test for quantized instance norm.
cc_binary(
    name = "quantized_instance_norm_test_android_only",
    testonly = 1,
    srcs = ["quantized_instance_norm_test.cc"],
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
    ] + select({
        "//tensorflow:android": [
            ":android_tensorflow_kernels",
            "//tensorflow/core:android_tensorflow_lib",
            "//tensorflow/core:android_tensorflow_test_lib",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:tensor_testutil",
        ],
    }),
)

tf_cc_test(
    name = "quantized_instance_norm_test",
    size = "small",
    srcs = ["quantized_instance_norm_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantized_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "remote_fused_graph_ops",
    prefix = "remote_fused_graph_execute_op",
    deps = [
        ":remote_fused_graph_execute_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:remote_fused_graph_ops_op_lib",
    ],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "@gemmlowp",
    ],
)

cc_library(
    name = "remote_fused_graph_execute_utils",
    srcs = [
        "i_remote_fused_graph_ops_definitions.cc",
        "remote_fused_graph_execute_utils.cc",
    ],
    hdrs = [
        "i_remote_fused_graph_executor.h",
        "i_remote_fused_graph_ops_definitions.h",
        "remote_fused_graph_execute_utils.h",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "remote_fused_graph_execute_op_test_utils",
    testonly = 1,
    srcs = ["remote_fused_graph_execute_op_test_utils.cc"],
    hdrs = ["remote_fused_graph_execute_op_test_utils.h"],
    deps = [
        ":remote_fused_graph_execute_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
    ],
)

tf_cc_test(
    name = "remote_fused_graph_execute_utils_test",
    size = "small",
    srcs = [
        "remote_fused_graph_execute_utils_test.cc",
    ],
    deps = [
        ":remote_fused_graph_execute_op_test_utils",
        ":remote_fused_graph_execute_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:remote_fused_graph_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
    ],
)

tf_cc_test(
    name = "remote_fused_graph_ops_test",
    size = "small",
    srcs = [
        "remote_fused_graph_execute_op_test.cc",
    ],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":remote_fused_graph_execute_op_test_utils",
        ":remote_fused_graph_execute_utils",
        ":remote_fused_graph_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:remote_fused_graph_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "remote_fused_graph_rewriter_transform",
    srcs = [
        "remote_fused_graph_rewriter_transform.cc",
    ],
    deps = [
        ":remote_fused_graph_execute_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:remote_fused_graph_ops",
        "//tensorflow/tools/graph_transforms:transform_utils",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "remote_fused_graph_rewriter_transform_test",
    size = "small",
    srcs = ["remote_fused_graph_rewriter_transform_test.cc"],
    deps = [
        ":remote_fused_graph_execute_op_test_utils",
        ":remote_fused_graph_execute_utils",
        ":remote_fused_graph_rewriter_transform",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/tools/graph_transforms:transform_utils",
    ],
)

tf_mkl_kernel_library(
    name = "mkl_conv_op",
    hdrs = [
        "mkl_quantized_conv_ops.h",
        "no_op.h",
    ],
    prefix = "mkl_conv",
    deps = [
        ":bounds_check",
        ":conv_ops",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:mkl_nn_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
    ] + mkl_deps(),
)

tf_cc_test_mkl(
    name = "mkl_conv_ops_test",
    size = "small",
    srcs = ["mkl_conv_ops_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_mkl_kernel_library(
    name = "mkl_tfconv_op",
    prefix = "mkl_tfconv",
    deps = [
        ":bounds_check",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:mkl_nn_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
    ] + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_input_conversion_op",
    hdrs = ["mkl_tfconv_op.h"],
    prefix = "mkl_input_conversion",
    deps = [
        ":bounds_check",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:mkl_nn_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
    ] + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_pooling_ops",
    srcs = [
        "mkl_avgpooling_op.cc",
        "mkl_maxpooling_op.cc",
        "mkl_pooling_ops_common.cc",
    ],
    hdrs = ["mkl_pooling_ops_common.h"],
    deps = [
        ":bounds_check",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:mkl_nn_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
    ] + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_relu_op",
    prefix = "mkl_relu",
    deps = [
        ":bounds_check",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:mkl_nn_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//third_party/eigen3",
    ] + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_softmax_op",
    prefix = "mkl_softmax",
    deps = [
        ":bounds_check",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:mkl_nn_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//third_party/eigen3",
    ] + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_fused_batch_norm_op",
    srcs = ["mkl_fused_batch_norm_op.cc"],
    deps = NN_DEPS + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_aggregate_ops",
    prefix = "mkl_aggregate_ops",
    deps = MATH_DEPS + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_concat_op",
    prefix = "mkl_concat_op",
    deps = [":quantization_utils"] + ARRAY_DEPS + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_reshape_op",
    prefix = "mkl_reshape_op",
    deps = ARRAY_DEPS + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_slice_op",
    prefix = "mkl_slice_op",
    deps = ARRAY_DEPS + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_identity_op",
    prefix = "mkl_identity_op",
    deps = ARRAY_DEPS + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_lrn_op",
    prefix = "mkl_lrn_op",
    deps = NN_DEPS + mkl_deps(),
)

tf_mkl_kernel_library(
    name = "mkl_cwise_ops_common",
    hdrs = [
        "cwise_ops.h",
        "cwise_ops_common.h",
        "cwise_ops_gradients.h",
    ],
    prefix = "mkl_cwise_ops_common",
    deps = NN_DEPS + mkl_deps() + [":cwise_op"],
)

tf_mkl_kernel_library(
    name = "mkl_transpose_op",
    srcs = [
        "mkl_transpose_op.cc",
    ],
    hdrs = ["transpose_op.h"],
    deps = ARRAY_DEPS + mkl_deps(),
)

# NOTE(lespeholt): This rule is deprecated, please use:
# tensorflow/core/util/batch_util.h
cc_library(
    name = "batch_util",
    hdrs = ["batch_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "boosted_trees_ops",
    deps = [
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_ops",
    ],
)

tf_kernel_library(
    name = "dataset_ops",
    deps = [
        "//tensorflow/core/kernels/data",
    ],
)

cc_library(
    name = "summary_interface",
    hdrs = ["summary_interface.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "summary_kernels",
    srcs = ["summary_kernels.cc"],
    deps = [
        "//tensorflow/contrib/tensorboard/db:schema",
        "//tensorflow/contrib/tensorboard/db:summary_db_writer",
        "//tensorflow/contrib/tensorboard/db:summary_file_writer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:summary_ops_op_lib",
        "//tensorflow/core/lib/db:sqlite",
    ],
)

tf_kernel_library(
    name = "decode_proto_op",
    srcs = [
        "decode_proto_op.cc",
    ],
    deps = [
        "//tensorflow/core:decode_proto_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/proto:decode",
        "//tensorflow/core/util/proto:descriptors",
        "//tensorflow/core/util/proto:proto_utils",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "encode_proto_op",
    srcs = ["encode_proto_op.cc"],
    deps = [
        "//tensorflow/core:encode_proto_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/proto:descriptors",
        "//tensorflow/core/util/proto:proto_utils",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "rpc_op",
    srcs = [
        "rpc_op.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:rpc_ops_op_lib",
        "//tensorflow/core/util/rpc:call_container",
        "//tensorflow/core/util/rpc:rpc_factory",
        "//tensorflow/core/util/rpc:rpc_factory_registry",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "unicode_script_op",
    srcs = ["unicode_script_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:string_ops_op_lib",
        "@icu//:common",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

# Library to link with when compiling the cwise_op kernels directly,
# e.g. for selective registration.
# should not be linked by projects that also link the cwise_op library.
cc_library(
    name = "cwise_lib",
    srcs = [
        "cwise_ops_common.cc",
        "meta_support.cc",
    ],
    hdrs = [
        "cwise_ops.h",
        "cwise_ops_common.h",
        "cwise_ops_gpu_common.cu.h",
        "cwise_ops_gpu_gradients.cu.h",
        "cwise_ops_gradients.h",
        "meta_support.h",
    ],
    deps = [
        ":bounds_check",
        ":quantization_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
        "@gemmlowp",
    ],
)

# Header-only version of cwise_lib for clients that want to use the cwise_ops
# functionality in their own custom ops.
cc_header_only_library(
    name = "cwise_lib_hdrs",
    deps = [
        ":cwise_lib",
    ],
)
