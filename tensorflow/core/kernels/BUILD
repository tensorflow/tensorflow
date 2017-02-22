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
        "//learning/deepmind/...",
        "//tensorflow/...",
    ],
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_copts",
    "tf_opts_nortti_if_android",
    "tf_kernel_library",
    "cc_header_only_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_tests")
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_kernel_tests_linkstatic",
)

config_setting(
    # Add "--define tensorflow_xsmm=1" to your build command to use libxsmm for
    # convolutions (and possibly more in the future). You will also need
    # appropriate -mavx*, as required by specific op you use.
    name = "xsmm",
    values = {
        "define": "tensorflow_xsmm=1",
    },
)

config_setting(
    # Add "--define tensorflow_xsmm_backward=1" to your build command to use
    # libxsmm for backward convolutions (and possibly more in the future). You
    # will also need appropriate -mavx*, as required by specific op you use.
    name = "xsmm_backward",
    values = {
        "define": "tensorflow_xsmm=1",
        "define": "tensorflow_xsmm_backward=1",
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
        "dense_update_ops.h",
        "slice_op.h",
        "strided_slice_op.h",
        "strided_slice_op_impl.h",
    ],
    gpu_srcs = [
        "dense_update_ops.h",
        "slice_op.h",
        "strided_slice_op.h",
        "strided_slice_op_impl.h",
        "strided_slice_op_gpu.cu.cc",
        "slice_op_gpu.cu.cc",
    ],
    deps = [
        ":bounds_check",
        ":ops_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
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
    deps = [
        ":eigen_helpers",
        ":ops_util_hdrs",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "conv_2d",
    hdrs = ["conv_2d.h"],
    deps = [
        ":eigen_helpers",
        "//tensorflow/core:framework",
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

cc_library(
    name = "conv_3d",
    hdrs = ["conv_3d.h"],
    deps = [
        ":eigen_helpers",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "fill_functor",
    srcs = ["fill_functor.cc"],
    hdrs = ["fill_functor.h"],
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
    ],
)

cc_library(
    name = "ops_testutil",
    testonly = 1,
    hdrs = ["ops_testutil.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "ops_util",
    srcs = ["ops_util.cc"],
    hdrs = ["ops_util.h"],
    copts = ["-Wno-sign-compare"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "ops_util_hdrs",
    hdrs = ["ops_util.h"],
    deps = [
        ":eigen_helpers",
        "//third_party/eigen3",
    ],
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

tf_kernel_library(
    name = "stage_op",
    srcs = ["stage_op.cc"],
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
    ],
)

cc_library(
    name = "queue_op",
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
        ":queue_op",
        ":typed_queue",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
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
    copts = ["-Wno-sign-compare"],
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
    name = "typed_queue",
    hdrs = ["typed_queue.h"],
    deps = [
        ":queue_base",
    ],
)

# Private support libraries ---------------------------------------------------

cc_library(
    name = "bounds_check",
    hdrs = ["bounds_check.h"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

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
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "eigen_helpers",
    hdrs = [
        "eigen_activations.h",
        "eigen_attention.h",
        "eigen_backward_cuboid_convolutions.h",
        "eigen_backward_spatial_convolutions.h",
        "eigen_cuboid_convolution.h",
        "eigen_patch_3d.h",
        "eigen_pooling.h",
        "eigen_softmax.h",
        "eigen_spatial_convolutions.h",
    ],
    deps = [
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "image_resizer_state",
    hdrs = ["image_resizer_state.h"],
    visibility = ["//visibility:private"],
    deps = [
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
]

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
        ":concat_op",
        ":constant_op",
        ":depth_space_ops",
        ":diag_op",
        ":edit_distance_op",
        ":extract_image_patches_op",
        ":gather_nd_op",
        ":gather_op",
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
        ":shape_ops",
        ":slice_op",
        ":split_op",
        ":split_v_op",
        ":strided_slice_op",
        ":tile_ops",
        ":transpose_op",
        ":unique_op",
        ":unpack_op",
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
    name = "concat_op",
    prefix = "concat_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "constant_op",
    prefix = "constant_op",
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
    name = "listdiff_op",
    prefix = "listdiff_op",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "matrix_band_part_op",
    prefix = "matrix_band_part_op",
    deps = ARRAY_DEPS,
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
    deps = ARRAY_DEPS,
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
    deps = ARRAY_DEPS,
)

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
    name = "inplace_ops",
    prefix = "inplace_ops",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "tile_ops",
    prefix = "tile_ops",
    deps = ARRAY_DEPS,
)

tf_kernel_library(
    name = "transpose_op",
    prefix = "transpose_op",
    deps = ARRAY_DEPS,
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
    name = "where_op",
    prefix = "where_op",
    deps = ARRAY_DEPS,
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

tf_cc_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.cc"],
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
        ":xsmm": ["xsmm_conv2d_test.cc"],
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
    ],
)

tf_cc_test(
    name = "conv_ops_test",
    size = "small",
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

tf_cc_test(
    name = "example_parsing_ops_test",
    size = "large",
    srcs = ["example_parsing_ops_test.cc"],
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
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

tf_cuda_cc_test(
    name = "gather_op_test",
    size = "small",
    srcs = ["gather_op_test.cc"],
    deps = [
        ":gather_op",
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
    name = "debug_ops_test",
    size = "small",
    srcs = ["debug_ops_test.cc"],
    deps = [
        ":debug_ops",
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
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
    alwayslink = 0,
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
    deps = DATA_FLOW_DEPS,
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

tf_kernel_library(
    name = "stack_ops",
    prefix = "stack_ops",
    deps = DATA_FLOW_DEPS,
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
    deps = DYNAMIC_DEPS,
)

tf_kernel_library(
    name = "dynamic_stitch_op",
    prefix = "dynamic_stitch_op",
    deps = DYNAMIC_DEPS,
)

LOOKUP_DEPS = [
    ":bounds_check",
    ":initializable_lookup_table",
    ":lookup_util",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:data_flow_ops_op_lib",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
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

tf_cc_tests(
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
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "fifo_queue",
    srcs = ["fifo_queue.cc"],
    hdrs = ["fifo_queue.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":queue_base",
        ":typed_queue",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
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
    deps = [
        ":bounds_check",
        ":dense_update_ops",
        ":gather_functor",
        ":scatter_functor",
        ":state",
        ":variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:resource_variable_ops_op_lib",
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
        ":decode_gif_op",
        ":decode_jpeg_op",
        ":decode_png_op",
        ":draw_bounding_box_op",
        ":encode_jpeg_op",
        ":encode_png_op",
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
    "//tensorflow/core:protos_all_cc",
]

tf_kernel_library(
    name = "adjust_contrast_op",
    prefix = "adjust_contrast_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "adjust_hue_op",
    prefix = "adjust_hue_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "adjust_saturation_op",
    prefix = "adjust_saturation_op",
    deps = IMAGE_DEPS,
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
    name = "decode_jpeg_op",
    prefix = "decode_jpeg_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "decode_png_op",
    prefix = "decode_png_op",
    deps = IMAGE_DEPS,
)

tf_kernel_library(
    name = "decode_gif_op",
    prefix = "decode_gif_op",
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
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
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

cc_library(
    name = "io",
    deps = [
        ":fixed_length_record_reader_op",
        ":identity_reader_op",
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
        ":matrix_inverse_op",
        ":matrix_solve_ls_op",
        ":matrix_solve_op",
        ":matrix_triangular_solve_op",
        ":qr_op",
        ":self_adjoint_eig_op",
        ":self_adjoint_eig_v2_op",
        ":svd_op",
    ],
)

LINALG_DEPS = [
    ":linalg_ops_common",
    "//third_party/eigen3",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:linalg_ops_op_lib",
]

tf_kernel_library(
    name = "cholesky_op",
    prefix = "cholesky_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "cholesky_grad",
    prefix = "cholesky_grad",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "determinant_op",
    prefix = "determinant_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "self_adjoint_eig_op",
    prefix = "self_adjoint_eig_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "self_adjoint_eig_v2_op",
    prefix = "self_adjoint_eig_v2_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_inverse_op",
    prefix = "matrix_inverse_op",
    deps = LINALG_DEPS,
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
    name = "matrix_triangular_solve_op",
    prefix = "matrix_triangular_solve_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "qr_op",
    prefix = "qr_op",
    deps = LINALG_DEPS,
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
    deps = LOGGING_DEPS,
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
    ],
)

MATH_DEPS = [
    ":bounds_check",
    ":fill_functor",
    ":transpose_functor",
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
        ":cast_op",
        ":check_numerics_op",
        ":cross_op",
        ":cwise_op",
        ":fft_ops",
        ":matmul_op",
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
    prefix = "batch_matmul_op",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "betainc_op",
    prefix = "betainc_op",
    deps = MATH_DEPS,
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
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "fft_ops",
    prefix = "fft_ops",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "matmul_op",
    defines = select({
        ":xsmm": [
            "TENSORFLOW_USE_LIBXSMM",
            "EIGEN_USE_LIBXSMM",
        ],
        "//conditions:default": [],
    }),
    prefix = "matmul_op",
    deps = MATH_DEPS + select({
        ":xsmm": [
            "@libxsmm_archive//:xsmm_avx",
        ],
        "//conditions:default": [],
    }),
)

tf_kernel_library(
    name = "reduction_ops",
    prefix = "reduction_ops",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "segment_reduction_ops",
    prefix = "segment_reduction_ops",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "scan_ops",
    prefix = "scan_ops",
    deps = MATH_DEPS,
)

tf_kernel_library(
    name = "sequence_ops",
    prefix = "sequence_ops",
    deps = MATH_DEPS,
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
        ":sparse_reduce_sum_op",
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
    name = "reduction_ops_test",
    size = "small",
    srcs = ["reduction_ops_test.cc"],
    linkopts = select({
        "//tensorflow:darwin": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    deps = [
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
        ":xsmm": ["xsmm_conv2d.cc"],
        "//conditions:default": [],
    }),
    hdrs = [
        "conv_grad_ops.h",
        "deep_conv2d.h",
        "gemm_functors.h",
        "winograd_transform.h",
    ] + select({
        ":xsmm": ["xsmm_conv2d.h"],
        "//conditions:default": [],
    }),
    defines = select({
        ":xsmm": [
            "TENSORFLOW_USE_LIBXSMM",
            "EIGEN_USE_LIBXSMM",
        ],
        "//conditions:default": [],
    }) + select({
        ":xsmm_backward": ["TENSORFLOW_USE_LIBXSMM_BACKWARD"],
        "//conditions:default": [],
    }),
    prefix = "conv_ops",
    deps = [
        ":bounds_check",
        ":conv_2d",
        ":conv_3d",
        ":image_resizer_state",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_ops_op_lib",
    ] + select({
        ":xsmm": [
            "@libxsmm_archive//:xsmm_avx",
        ],
        "//conditions:default": [],
    }),
)

tf_kernel_library(
    name = "depthwise_conv_op",
    prefix = "depthwise_conv_op",
    deps = [
        ":conv_ops",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
    ],
)

tf_kernel_library(
    name = "depthwise_conv_grad_op",
    hdrs = [
        "depthwise_conv_op.h",
    ],
    prefix = "depthwise_conv_grad_op",
    deps = [
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
    ],
)

cc_library(
    name = "nn",
    deps = [
        ":batch_norm_op",
        ":bias_op",
        ":conv_ops",
        ":depthwise_conv_grad_op",
        ":depthwise_conv_op",
        ":dilation_ops",
        ":fused_batch_norm_op",
        ":in_topk_op",
        ":l2loss_op",
        ":lrn_op",
        ":relu_op",
        ":softmax_op",
        ":softplus_op",
        ":softsign_op",
        ":topk_op",
        ":xent_op",
    ],
)

NN_DEPS = [
    ":bounds_check",
    ":conv_2d",
    ":fused_batch_norm_util_gpu",
    ":ops_util",
    ":pooling_ops",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:nn_grad",
    "//tensorflow/core:nn_ops_op_lib",
    "//third_party/eigen3",
]

tf_kernel_library(
    name = "batch_norm_op",
    prefix = "batch_norm_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "bias_op",
    prefix = "bias_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "fused_batch_norm_op",
    prefix = "fused_batch_norm_op",
    deps = NN_DEPS,
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
    deps = NN_DEPS,
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
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "xent_op",
    prefix = "xent_op",
    deps = NN_DEPS,
)

tf_kernel_library(
    name = "l2loss_op",
    prefix = "l2loss_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_grad",
        "//tensorflow/core:nn_ops_op_lib",
        "//third_party/eigen3",
    ],
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
        "pooling_ops_common.h",
    ],
    gpu_srcs = [
        "avgpooling_op.h",
        "avgpooling_op_gpu.cu.cc",
        "maxpooling_op.h",
        "maxpooling_op_gpu.cu.cc",
        "maxpooling_op_gpu.h",
        "pooling_ops_common.h",
        "pooling_ops_common_gpu.h",
    ],
    deps = [
        ":conv_2d",
        ":conv_3d",
        ":conv_ops",
        ":eigen_helpers",
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
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
    visibility = ["//visibility:private"],
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
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "parsing",
    deps = [
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
    name = "example_parsing_ops",
    prefix = "example_parsing_ops",
    deps = PARSING_DEPS,
)

tf_kernel_library(
    name = "parse_tensor_op",
    prefix = "parse_tensor_op",
    deps = PARSING_DEPS,
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
        ":random_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
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

cc_library(
    name = "sparse",
    deps = [
        ":serialize_sparse_op",
        ":sparse_add_grad_op",
        ":sparse_add_op",
        ":sparse_concat_op",
        ":sparse_dense_binary_op_shared",
        ":sparse_reduce_sum_op",
        ":sparse_reorder_op",
        ":sparse_reshape_op",
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
    ":bounds_check",
    ":cwise_op",
    ":fill_functor",
    ":scatter_functor",
    "//third_party/eigen3",
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
    name = "sparse_reduce_sum_op",
    prefix = "sparse_reduce_sum_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_dense_binary_op_shared",
    prefix = "sparse_dense_binary_op_shared",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_sparse_binary_op_shared",
    prefix = "sparse_sparse_binary_op_shared",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_reorder_op",
    prefix = "sparse_reorder_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_reshape_op",
    prefix = "sparse_reshape_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_softmax",
    prefix = "sparse_softmax",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_split_op",
    prefix = "sparse_split_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_tensor_dense_add_op",
    prefix = "sparse_tensor_dense_add_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_tensor_dense_matmul_op",
    prefix = "sparse_tensor_dense_matmul_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_to_dense_op",
    prefix = "sparse_to_dense_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "sparse_xent_op",
    prefix = "sparse_xent_op",
    deps = SPARSE_DEPS,
)

tf_kernel_library(
    name = "serialize_sparse_op",
    prefix = "serialize_sparse_op",
    deps = SPARSE_DEPS,
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
        "smooth-hinge-loss.h",
        "squared-loss.h",
    ],
    deps = ["//tensorflow/core:framework_headers_lib"],
)

cc_test(
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
]

tf_kernel_library(
    name = "count_up_to_op",
    prefix = "count_up_to_op",
    deps = STATE_DEPS,
)

tf_kernel_library(
    name = "dense_update_ops",
    prefix = "dense_update_ops",
    deps = STATE_DEPS,
)

tf_kernel_library(
    name = "scatter_op",
    prefix = "scatter_op",
    deps = STATE_DEPS,
)

tf_kernel_library(
    name = "scatter_nd_op",
    prefix = "scatter_nd_op",
    deps = STATE_DEPS,
)

tf_kernel_library(
    name = "variable_ops",
    prefix = "variable_ops",
    deps = STATE_DEPS,
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
        ":string_join_op",
        ":string_split_op",
        ":string_to_hash_bucket_op",
        ":substr_op",
    ],
)

STRING_DEPS = [
    ":bounds_check",
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
    name = "string_join_op",
    prefix = "string_join_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "string_split_op",
    prefix = "string_split_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "substr_op",
    prefix = "substr_op",
    deps = STRING_DEPS,
)

tf_kernel_library(
    name = "as_string_op",
    prefix = "as_string_op",
    deps = STRING_DEPS,
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
        ":random_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cuda_cc_test(
    name = "multinomial_op_test",
    size = "small",
    srcs = ["multinomial_op_test.cc"],
    deps = [
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
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:word2vec_ops",
    ],
)

# Android libraries -----------------------------------------------------------

# Changes to the Android srcs here should be replicated in
# tensorflow/contrib/makefile/tf_op_files.txt
# LINT.IfChange
filegroup(
    name = "android_srcs",
    srcs = [
        "avgpooling_op.h",
        "bounds_check.h",
        "cwise_ops.h",
        "cwise_ops_common.h",
        "cwise_ops_gradients.h",
        "eigen_activations.h",
        "eigen_attention.h",
        "eigen_backward_cuboid_convolutions.h",
        "eigen_backward_spatial_convolutions.h",
        "eigen_cuboid_convolution.h",
        "eigen_patch_3d.h",
        "eigen_pooling.h",
        "eigen_softmax.h",
        "eigen_spatial_convolutions.h",
        "fifo_queue.h",
        "maxpooling_op.h",
        "ops_util.cc",
        "ops_util.h",
        "padding_fifo_queue.h",
        "pooling_ops_common.cc",
        "pooling_ops_common.h",
        "queue_base.h",
        "queue_op.h",
        "typed_queue.h",
    ],
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
        "dense_update_ops.cc",
        "dense_update_ops.h",
        "example_parsing_ops.cc",
        "fill_functor.cc",
        "fill_functor.h",
        "function_ops.cc",
        "gather_functor.h",
        "gather_op.cc",
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
        "softmax_op.h",
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
        "batch_norm_op.h",
        "control_flow_ops.h",
        "conv_2d.h",
        "conv_ops.h",
        "depthtospace_op.h",
        "depthwise_conv_op.h",
        "fake_quant_ops_functor.h",
        "gemm_functors.h",
        "image_resizer_state.h",
        "maxpooling_op.h",
        "mirror_pad_op.h",
        "mirror_pad_op_cpu_impl.h",
        "pad_op.h",
        "random_op.h",
        "reduction_ops.h",
        "reduction_ops_common.h",
        "relu_op.h",
        "relu_op_functor.h",
        "resize_bilinear_op.h",
        "reverse_op.h",
        "save_restore_tensor.h",
        "softplus_op.h",
        "softsign_op.h",
        "spacetobatch_functor.h",
        "tensor_array.h",
        "tile_ops_cpu_impl.h",
        "tile_ops_impl.h",
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
        "conv_ops_fused.cc",
        "conv_ops_using_gemm.cc",
        "crop_and_resize_op.cc",
        "crop_and_resize_op.h",
        "cwise_op_abs.cc",
        "cwise_op_add_1.cc",
        "cwise_op_add_2.cc",
        "cwise_op_div.cc",
        "cwise_op_equal_to_1.cc",
        "cwise_op_equal_to_2.cc",
        "cwise_op_exp.cc",
        "cwise_op_floor.cc",
        "cwise_op_greater.cc",
        "cwise_op_greater_equal.cc",
        "cwise_op_isfinite.cc",
        "cwise_op_less.cc",
        "cwise_op_log.cc",
        "cwise_op_logical_and.cc",
        "cwise_op_logical_not.cc",
        "cwise_op_maximum.cc",
        "cwise_op_minimum.cc",
        "cwise_op_mul_1.cc",
        "cwise_op_mul_2.cc",
        "cwise_op_neg.cc",
        "cwise_op_pow.cc",
        "cwise_op_reciprocal.cc",
        "cwise_op_rsqrt.cc",
        "cwise_op_select.cc",
        "cwise_op_sigmoid.cc",
        "cwise_op_sign.cc",
        "cwise_op_sqrt.cc",
        "cwise_op_square.cc",
        "cwise_op_squared_difference.cc",
        "cwise_op_sub.cc",
        "cwise_op_tanh.cc",
        "deep_conv2d.cc",
        "deep_conv2d.h",
        "depthwise_conv_op.cc",
        "dynamic_partition_op.cc",
        "fake_quant_ops.cc",
        "fifo_queue.cc",
        "fused_batch_norm_op.cc",
        "winograd_transform.h",
        ":android_extended_ops_headers",
    ] + select({
        ":xsmm": [
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
        "depthtospace_op.cc",
        "dynamic_stitch_op.cc",
        "in_topk_op.cc",
        "logging_ops.cc",
        "lrn_op.cc",
        "maxpooling_op.cc",
        "mirror_pad_op.cc",
        "mirror_pad_op_cpu_impl_1.cc",
        "mirror_pad_op_cpu_impl_2.cc",
        "mirror_pad_op_cpu_impl_3.cc",
        "mirror_pad_op_cpu_impl_4.cc",
        "mirror_pad_op_cpu_impl_5.cc",
        "pad_op.cc",
        "padding_fifo_queue.cc",
        "padding_fifo_queue_op.cc",
        "queue_base.cc",
        "queue_ops.cc",
        "random_op.cc",
        "reduction_ops_any.cc",
        "reduction_ops_common.cc",
        "reduction_ops_max.cc",
        "reduction_ops_mean.cc",
        "reduction_ops_min.cc",
        "reduction_ops_prod.cc",
        "reduction_ops_sum.cc",
        "relu_op.cc",
        "resize_bilinear_op.cc",
        "resize_nearest_neighbor_op.cc",
        "restore_op.cc",
        "reverse_op.cc",
        "save_op.cc",
        "save_restore_tensor.cc",
        "save_restore_v2_ops.cc",
        "session_ops.cc",
        "softplus_op.cc",
        "softsign_op.cc",
        "spacetobatch_functor.cc",
        "spacetobatch_op.cc",
        "sparse_to_dense_op.cc",
        "stack_ops.cc",
        "summary_op.cc",
        "tensor_array.cc",
        "tensor_array_ops.cc",
        "tile_ops.cc",
        "tile_ops_cpu_impl_1.cc",
        "tile_ops_cpu_impl_2.cc",
        "tile_ops_cpu_impl_3.cc",
        "tile_ops_cpu_impl_4.cc",
        "tile_ops_cpu_impl_5.cc",
        "tile_ops_cpu_impl_6.cc",
        "tile_ops_cpu_impl_7.cc",
        "topk_op.cc",
        "training_ops.cc",
        "transpose_functor_cpu.cc",
        "transpose_op.cc",
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
        "quantized_batch_norm_op.cc",
        "quantized_bias_add_op.cc",
        "quantized_concat_op.cc",
        "quantized_conv_ops.cc",
        "quantized_instance_norm.cc",
        "quantized_matmul_op.cc",
        "quantized_mul_op.cc",
        "quantized_pooling_ops.cc",
        "quantized_reshape_op.cc",
        "reference_gemm.h",
        "requantization_range_op.cc",
        "requantize.cc",
        "reshape_op.h",
    ],
    visibility = ["//visibility:public"],
)

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
            "*testutil*",
            "*testlib*",
            "*main.cc",
            "*_gpu*",
            "*_3d*",
            "*.cu.*",
            # Ops already in android_srcs
            "ops_util.cc",
            "pooling_ops_common.cc",
            # Ops which we are currently excluding because they are likely
            # not used on Android. Those ops also do not compile if included,
            # unless we add the additional deps they need.
            "tf_record_reader_op.*",
            "string_to_hash_bucket_op.*",
            "sdca_ops.*",
            "sdca_internal.*",
            "text_line_reader_op.*",
            "summary_image_op.*",
            "encode_png_op.*",
            "decode_png_op.*",
            "encode_jpeg_op.*",
            "decode_jpeg_op.*",
            "decode_gif_op.*",
            "identity_reader_op.*",
            "remote_fused_graph_execute_op.*",
            "fixed_length_record_reader_op.*",
            "whole_file_read_ops.*",
            "sample_distorted_bounding_box_op.*",
            "ctc_loss_op.*",
            # Excluded due to experimental status:
            "debug_ops.*",
            "scatter_nd_op*",
            # Lib CURL is not supported on Android.
            "bigquery*",
        ],
    ),
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
    tags = [
        "manual",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:android_tensorflow_lib_lite",
        "//tensorflow/core:protos_cc",
        "//third_party/eigen3",
        "@gemmlowp//:gemmlowp",
    ],
    alwayslink = 1,
)

#   Quantization-specific OpKernels

tf_kernel_library(
    name = "quantized_ops",
    srcs = [
        "dequantize_op.cc",
        "meta_support.cc",
        "quantization_utils.cc",
        "quantize_down_and_shrink_range.cc",
        "quantize_op.cc",
        "quantized_activation_ops.cc",
        "quantized_batch_norm_op.cc",
        "quantized_bias_add_op.cc",
        "quantized_concat_op.cc",
        "quantized_conv_ops.cc",
        "quantized_instance_norm.cc",
        "quantized_matmul_op.cc",
        "quantized_mul_op.cc",
        "quantized_pooling_ops.cc",
        "quantized_reshape_op.cc",
        "requantization_range_op.cc",
        "requantize.cc",
        "reshape_op.h",
    ],
    hdrs = [
        "meta_support.h",
        "quantization_utils.h",
        "reference_gemm.h",
    ],
    deps = [
        ":concat_lib_hdrs",
        ":conv_ops",
        ":eigen_helpers",
        ":ops_util",
        ":pooling_ops",
        "//tensorflow/core",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//third_party/eigen3",
        "@gemmlowp//:gemmlowp",
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
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "quantized_activation_ops_test",
    srcs = ["quantized_activation_ops_test.cc"],
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
    name = "quantized_bias_add_op_test",
    size = "small",
    srcs = ["quantized_bias_add_op_test.cc"],
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
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["quantized_conv_ops_test.cc"],
    tags = ["nomsan"],  # http://b/32242946
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

# Android-only test for quantized instance norm.
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
            "//tensorflow/core:framework",
            "//tensorflow/core:tensor_testutil",
            "//tensorflow/core:test_main",
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
        "//tensorflow/core:test_main",
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
            "//tensorflow/core:test_main",
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
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "remote_fused_graph_ops",
    prefix = "remote_fused_graph_ops",
    deps = [
        ":remote_fused_graph_execute_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:remote_fused_graph_ops_op_lib",
        "//tensorflow/core/kernels/hexagon:graph_transferer",
    ],
)

cc_library(
    name = "remote_fused_graph_execute_op",
    srcs = ["remote_fused_graph_execute_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/hexagon:graph_transferer",
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
        ":remote_fused_graph_execute_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:remote_fused_graph_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

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
