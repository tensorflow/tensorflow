# Description:
# Op kernel implementations for TensorFlow.
#
# Note: Any test that uses GPU support and which we would like to
# benchmark should be linked statically so that it can be executed
# from a py_binary or cuda_py_test test logger.  For such a test,
# append "_gpu" to the test name to invoke the GPU benchmarks.  Example:
#
#   # for CPU tests
#   $ bazel test -c opt --copt=-mavx //third_party/tensorflow/core/kernels:my_op_test
#   # for GPU benchmarks
#   $ bazel run -c opt --copt=-mavx --config=cuda //third_party/tensorflow/core/kernels:my_op_test_gpu -- --benchmarks=..
#
package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

package_group(
    name = "friends",
    packages = [
        "//learning/deepmind/tensorflow/...",
        "//tensorflow/...",
    ],
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_opts_nortti_if_android",
    "tf_kernel_libraries",
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
        "strided_slice_op_inst_1.cc",
        "strided_slice_op_inst_2.cc",
        "strided_slice_op_inst_3.cc",
        "strided_slice_op_inst_4.cc",
        "strided_slice_op_inst_5.cc",
        "strided_slice_op_inst_6.cc",
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
    srcs = ["concat_lib_cpu.cc"],
    hdrs = [
        "concat_lib.h",
        "concat_lib_cpu.h",
    ],
    gpu_srcs = [
        "concat_lib_gpu.cu.cc",
        "concat_lib.h",
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
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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

tf_proto_library(
    name = "reader_base_proto",
    srcs = ["reader_base.proto"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
)

cc_library(
    name = "reader_base",
    srcs = ["reader_base.cc"],
    hdrs = ["reader_base.h"],
    deps = [
        ":reader_base_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "save_restore_tensor",
    srcs = ["save_restore_tensor.cc"],
    hdrs = ["save_restore_tensor.h"],
    deps = [
        ":bounds_check",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
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

tf_kernel_libraries(
    name = "array",
    prefixes = [
        "bcast_ops",
        "bitcast_op",
        "concat_op",
        "constant_op",
        "debug_ops",
        "diag_op",
        "matrix_band_part_op",
        "matrix_diag_op",
        "matrix_set_diag_op",
        "edit_distance_op",
        "gather_op",
        "gather_nd_op",
        "identity_op",
        "immutable_constant_op",
        "listdiff_op",
        "mirror_pad_op",
        "one_hot_op",
        "pack_op",
        "pad_op",
        "quantize_and_dequantize_op",
        "reshape_op",
        "reverse_op",
        "reverse_sequence_op",
        "shape_ops",
        "slice_op",
        "split_op",
        "tile_ops",
        "transpose_op",
        "unique_op",
        "unpack_op",
        "where_op",
    ],
    deps = [
        ":batch_space_ops",
        ":bounds_check",
        ":concat_lib",
        ":cuda_device_array",
        ":depth_space_ops",
        ":extract_image_patches_op",
        ":fill_functor",
        ":gather_functor",
        ":ops_util",
        ":split_lib",
        ":strided_slice_op",
        ":transpose_functor",
        "//tensorflow/core:array_grad",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/debug:debug_io_utils",
        "//third_party/eigen3",
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

cc_library(
    name = "gather_functor_cpu_header",
    hdrs = ["gather_functor.h"],
    visibility = [":friends"],
    deps = [
        ":bounds_check",
        "//tensorflow/core:framework_lite",
        "//third_party/eigen3",
    ],
)

tf_cuda_library(
    name = "gather_functor",
    hdrs = [
        "gather_functor.h",
        "gather_functor_gpu.cu.h",
    ],
    visibility = [":friends"],
    deps = [
        ":bounds_check",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
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

tf_cc_test(
    name = "quantize_and_dequantize_op_test",
    size = "small",
    srcs = ["quantize_and_dequantize_op_test.cc"],
    deps = [
        ":ops_testutil",
        ":ops_util",
        ":quantize_and_dequantize_op",
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

tf_kernel_libraries(
    name = "data_flow",
    prefixes = [
        "dynamic_partition_op",
        "dynamic_stitch_op",
        "barrier_ops",
        "fifo_queue_op",
        "priority_queue_op",
        "lookup_table_init_op",
        "lookup_table_op",
        "padding_fifo_queue_op",
        "queue_ops",
        "random_shuffle_queue_op",
        "session_ops",
        "stack_ops",
        "tensor_array_ops",
    ],
    deps = [
        ":bounds_check",
        ":concat_lib",
        ":fifo_queue",
        ":initializable_lookup_table",
        ":lookup_util",
        ":padding_fifo_queue",
        ":priority_queue",
        ":queue_base",
        ":queue_op",
        ":split_lib",
        ":tensor_array",
        ":typed_queue",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:data_flow_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/eigen3",
    ],
)

tf_cc_tests(
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

tf_kernel_libraries(
    name = "image",
    prefixes = [
        "adjust_contrast_op",
        "colorspace_op",
        "crop_and_resize_op",
        "decode_jpeg_op",
        "decode_png_op",
        "decode_gif_op",
        "draw_bounding_box_op",
        "encode_jpeg_op",
        "attention_ops",
        "encode_png_op",
        "non_max_suppression_op",
        "random_crop_op",
        "resize_area_op",
        "resize_bicubic_op",
        "resize_bilinear_op",
        "resize_nearest_neighbor_op",
        "sample_distorted_bounding_box_op",
    ],
    deps = [
        ":bounds_check",
        ":eigen_helpers",
        ":image_resizer_state",
        "//tensorflow/core:framework",
        "//tensorflow/core:image_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
)

tf_cc_tests(
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
    srcs = [
        "adjust_contrast_op_test.cc",
        "colorspace_op_test.cc",
        "crop_and_resize_op_test.cc",
        "non_max_suppression_op_test.cc",
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

tf_kernel_libraries(
    name = "io",
    prefixes = [
        "fixed_length_record_reader_op",
        "identity_reader_op",
        "matching_files_op",
        "reader_ops",
        "restore_op",
        "save_op",
        "save_restore_v2_ops",
        "text_line_reader_op",
        "tf_record_reader_op",
        "whole_file_read_ops",
    ],
    deps = [
        ":bounds_check_lib",
        ":ops_util",
        ":reader_base",
        ":save_restore_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:io_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

tf_cc_tests(
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

tf_kernel_libraries(
    name = "linalg",
    prefixes = [
        "cholesky_op",
        "cholesky_grad",
        "determinant_op",
        "self_adjoint_eig_op",
        "self_adjoint_eig_v2_op",
        "matrix_inverse_op",
        "matrix_solve_ls_op",
        "matrix_solve_op",
        "matrix_triangular_solve_op",
        "svd_op",
    ],
    deps = [
        ":linalg_ops_common",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:linalg_ops_op_lib",
        "//third_party/eigen3",
    ],
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

tf_kernel_libraries(
    name = "logging",
    prefixes = [
        "logging_ops",
        "summary_audio_op",
        "summary_image_op",
        "summary_op",
        "summary_tensor_op",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:logging_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_tests(
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

tf_kernel_libraries(
    name = "math",
    prefixes = [
        "aggregate_ops",
        "argmax_op",
        "batch_matmul_op",
        "betainc_op",
        "cast_op",
        "check_numerics_op",
        "cross_op",
        "cwise_op",
        "fft_ops",
        "matmul_op",
        "reduction_ops",
        "segment_reduction_ops",
        "scan_ops",
        "sequence_ops",
        "sparse_matmul_op",
    ],
    deps = [
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
        "conv_grad_ops.cc",
        "conv_grad_ops_3d.cc",
        "deep_conv2d.cc",
    ],
    hdrs = [
        "conv_grad_ops.h",
        "deep_conv2d.h",
        "gemm_functors.h",
        "winograd_transform.h",
    ],
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
        "//tensorflow/core:nn_ops_op_lib",
    ],
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

tf_kernel_libraries(
    name = "nn",
    prefixes = [
        "batch_norm_op",
        "bias_op",
        "fused_batch_norm_op",
        "in_topk_op",
        "l2loss_op",
        "lrn_op",
        "relu_op",
        "softmax_op",
        "softplus_op",
        "softsign_op",
        "topk_op",
        "xent_op",
    ],
    deps = [
        ":bounds_check",
        ":conv_2d",
        ":conv_ops",
        ":depthwise_conv_grad_op",
        ":depthwise_conv_op",
        ":dilation_ops",
        ":fused_batch_norm_util_gpu",
        ":ops_util",
        ":pooling_ops",
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

tf_kernel_libraries(
    name = "parsing",
    prefixes = [
        "decode_csv_op",
        "decode_raw_op",
        "example_parsing_ops",
        "parse_tensor_op",
        "string_to_number_op",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:parsing_ops_op_lib",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_libraries(
    name = "random_ops",
    prefixes = [
        "random_op",
        "random_shuffle_op",
    ],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:random_ops_op_lib",
    ],
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

tf_kernel_libraries(
    name = "required",
    prefixes = [
        "no_op",
        "sendrecv_ops",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:sendrecv_ops_op_lib",
    ],
)

tf_kernel_libraries(
    name = "sparse",
    prefixes = [
        "sparse_add_grad_op",
        "sparse_add_op",
        "sparse_concat_op",
        "sparse_reduce_sum_op",
        "sparse_dense_binary_op_shared",
        "sparse_sparse_binary_op_shared",
        "sparse_reorder_op",
        "sparse_reshape_op",
        "sparse_softmax",
        "sparse_split_op",
        "sparse_tensor_dense_add_op",
        "sparse_tensor_dense_matmul_op",
        "sparse_to_dense_op",
        "sparse_xent_op",
        "serialize_sparse_op",
    ],
    deps = [
        ":bounds_check",
        ":cwise_op",
        ":fill_functor",
        ":scatter_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:sparse_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_cuda_cc_tests(
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
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_kernel_library(
    name = "sdca_ops",
    prefix = "sdca_ops",
    deps = [
        ":loss_updaters",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:sdca_ops_op_lib",
        "//third_party/eigen3",
        "@farmhash_archive//:farmhash",
    ],
    alwayslink = 1,
)

tf_kernel_libraries(
    name = "state",
    prefixes = [
        "count_up_to_op",
        "dense_update_ops",
        "scatter_op",
        "variable_ops",
    ],
    deps = [
        ":assign_op",
        ":bounds_check",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:state_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "scatter_op_test",
    size = "small",
    srcs = ["scatter_op_test.cc"],
    deps = [
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

tf_kernel_libraries(
    name = "string",
    prefixes = [
        "string_to_hash_bucket_op",
        "reduce_join_op",
        "string_join_op",
        "string_split_op",
        "as_string_op",
        "base64_ops",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:string_ops_op_lib",
    ],
)

tf_kernel_library(
    name = "training_ops",
    prefix = "training_ops",
    deps = [
        ":bounds_check",
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

# Android libraries -----------------------------------------------------------

# Changes to the Android srcs here should be replicated in
# tensorflow/contrib/makefile/tf_op_files.txt
# LINT.IfChange
filegroup(
    name = "android_srcs",
    srcs = [
        "avgpooling_op.h",
        "bounds_check.h",
        "eigen_activations.h",
        "eigen_attention.h",
        "eigen_backward_cuboid_convolutions.h",
        "eigen_backward_spatial_convolutions.h",
        "eigen_cuboid_convolution.h",
        "eigen_patch_3d.h",
        "eigen_pooling.h",
        "eigen_softmax.h",
        "eigen_spatial_convolutions.h",
        "maxpooling_op.h",
        "ops_util.cc",
        "ops_util.h",
        "pooling_ops_common.cc",
        "pooling_ops_common.h",
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
        "gather_op.h",
        "identity_op.cc",
        "identity_op.h",
        "immutable_constant_op.cc",
        "immutable_constant_op.h",
        "matmul_op.cc",
        "matmul_op.h",
        "no_op.cc",
        "no_op.h",
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
        "slice_op.cc",
        "slice_op.h",
        "slice_op_cpu_impl.h",
        "slice_op_cpu_impl_1.cc",
        "slice_op_cpu_impl_2.cc",
        "slice_op_cpu_impl_3.cc",
        "slice_op_cpu_impl_4.cc",
        "slice_op_cpu_impl_5.cc",
        "slice_op_cpu_impl_6.cc",
        "softmax_op.cc",
        "softmax_op.h",
        "softmax_op_functor.h",
        "split_lib.h",
        "split_lib_cpu.cc",
        "split_op.cc",
        "strided_slice_op.cc",
        "strided_slice_op.h",
        "strided_slice_op_impl.h",
        "strided_slice_op_inst_1.cc",
        "strided_slice_op_inst_2.cc",
        "strided_slice_op_inst_3.cc",
        "strided_slice_op_inst_4.cc",
        "strided_slice_op_inst_5.cc",
        "strided_slice_op_inst_6.cc",
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
        "depthwise_conv_op.h",
        "image_resizer_state.h",
        "maxpooling_op.h",
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
        "conv_grad_ops.cc",
        "conv_grad_ops.h",
        "conv_ops.cc",
        "cwise_op_abs.cc",
        "cwise_op_add_1.cc",
        "cwise_op_add_2.cc",
        "cwise_op_div.cc",
        "cwise_op_equal_to_1.cc",
        "cwise_op_equal_to_2.cc",
        "cwise_op_exp.cc",
        "cwise_op_floor.cc",
        "cwise_op_greater.cc",
        "cwise_op_inverse.cc",
        "cwise_op_isfinite.cc",
        "cwise_op_less.cc",
        "cwise_op_log.cc",
        "cwise_op_maximum.cc",
        "cwise_op_minimum.cc",
        "cwise_op_mul_1.cc",
        "cwise_op_mul_2.cc",
        "cwise_op_neg.cc",
        "cwise_op_rsqrt.cc",
        "cwise_op_select.cc",
        "cwise_op_sigmoid.cc",
        "cwise_op_sqrt.cc",
        "cwise_op_square.cc",
        "cwise_op_squared_difference.cc",
        "cwise_op_sub.cc",
        "cwise_op_tanh.cc",
        "deep_conv2d.cc",
        "deep_conv2d.h",
        "depthwise_conv_op.cc",
        "dynamic_partition_op.cc",
        "fused_batch_norm_op.cc",
        "winograd_transform.h",
        ":android_extended_ops_headers",
    ],
)

filegroup(
    name = "android_extended_ops_group2",
    srcs = [
        "ctc_decoder_ops.cc",
        "dynamic_stitch_op.cc",
        "in_topk_op.cc",
        "lrn_op.cc",
        "maxpooling_op.cc",
        "pad_op.cc",
        "random_op.cc",
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
        "sparse_to_dense_op.cc",
        "stack_ops.cc",
        "summary_op.cc",
        "tile_ops.cc",
        "tile_ops_cpu_impl_1.cc",
        "tile_ops_cpu_impl_2.cc",
        "tile_ops_cpu_impl_3.cc",
        "tile_ops_cpu_impl_4.cc",
        "tile_ops_cpu_impl_5.cc",
        "tile_ops_cpu_impl_6.cc",
        "topk_op.cc",
        "training_ops.cc",
        "transpose_functor_cpu.cc",
        "transpose_op.cc",
        "where_op.cc",
        "xent_op.cc",
        ":android_extended_ops_headers",
    ],
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
            "text_line_reader_op.*",
            "summary_image_op.*",
            "encode_png_op.*",
            "decode_png_op.*",
            "encode_jpeg_op.*",
            "decode_jpeg_op.*",
            "decode_gif_op.*",
            "identity_reader_op.*",
            "reader_base.*",
            "fixed_length_record_reader_op.*",
            "whole_file_read_ops.*",
            "sample_distorted_bounding_box_op.*",
            "ctc_loss_op.*",
            # Excluded due to experimental status:
            "debug_ops.*",
            # Ops excluded because they do not build correctly for Android.
            # See b/29213790
            "sparse_matmul_op.*",
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
    ],
    alwayslink = 1,
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
