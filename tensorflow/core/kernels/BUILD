# Description:
# Op kernel implementations for TensorFlow.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_kernel_libraries",
    "tf_kernel_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
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
    name = "concat_lib",
    srcs = ["concat_lib_cpu.cc"],
    hdrs = ["concat_lib.h"],
    gpu_srcs = [
        "concat_lib_gpu.cu.cc",
        "concat_lib.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
    alwayslink = 0,
)

cc_library(
    name = "conv_2d",
    hdrs = ["conv_2d.h"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "fill_functor",
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
    srcs = ["ops_testutil.cc"],
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

tf_cc_test(
    name = "ops_util_test",
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
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

# OpKernel libraries ----------------------------------------------------------

tf_kernel_libraries(
    name = "array",
    prefixes = [
        "bcast_ops",
        "bitcast_op",
        "depthtospace_op",
        "concat_op",
        "constant_op",
        "diag_op",
        "edit_distance_op",
        "gather_op",
        "identity_op",
        "listdiff_op",
        "one_hot_op",
        "pack_op",
        "pad_op",
        "random_shuffle_op",
        "reshape_op",
        "reverse_op",
        "reverse_sequence_op",
        "shape_ops",
        "slice_op",
        "spacetodepth_op",
        "split_op",
        "tile_ops",
        "transpose_op",
        "unique_op",
        "unpack_op",
        "where_op",
    ],
    deps = [
        ":bounds_check",
        ":concat_lib",
        ":fill_functor",
        ":ops_util",
        ":split_lib",
        ":transpose_functor",
        "//tensorflow/core:array_grad",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
)

tf_cc_tests(
    tests = [
        "concat_op_test",
        "constant_op_test",
        "gather_op_test",
        "identity_op_test",
        "reverse_op_test",
        "slice_op_test",
        "unique_op_test",
    ],
    deps = [
        ":array",
        ":ops_testutil",
        ":ops_util",
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
        "fifo_queue_op",
        "lookup_table_init_op",
        "lookup_table_op",
        "padding_fifo_queue_op",
        "queue_ops",
        "random_shuffle_queue_op",
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
    tests = [
        "dynamic_partition_op_test",
        "dynamic_stitch_op_test",
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
    name = "tensor_array",
    srcs = ["tensor_array.cc"],
    hdrs = ["tensor_array.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
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

tf_kernel_libraries(
    name = "io",
    prefixes = [
        "fixed_length_record_reader_op",
        "identity_reader_op",
        "matching_files_op",
        "reader_ops",
        "restore_op",
        "save_op",
        "text_line_reader_op",
        "tf_record_reader_op",
        "whole_file_read_ops",
    ],
    deps = [
        ":ops_util",
        ":reader_base",
        ":save_restore_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:io_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_tests(
    tests = [
        "restore_op_test",
        "save_op_test",
    ],
    deps = [
        ":io",
        ":ops_testutil",
        ":ops_util",
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

tf_kernel_libraries(
    name = "image",
    prefixes = [
        "adjust_contrast_op",
        "attention_ops",
        "colorspace_op",
        "decode_jpeg_op",
        "decode_png_op",
        "draw_bounding_box_op",
        "encode_jpeg_op",
        "encode_png_op",
        "random_crop_op",
        "resize_area_op",
        "resize_bicubic_op",
        "resize_bilinear_op",
        "resize_nearest_neighbor_op",
        "sample_distorted_bounding_box_op",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:image_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
)

tf_cc_tests(
    tests = [
        "adjust_contrast_op_benchmark_test",
        "adjust_contrast_op_test",
        "colorspace_op_test",
        "resize_bicubic_op_test",
        "resize_bilinear_op_test",
        "resize_nearest_neighbor_op_test",
    ],
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
    name = "linalg",
    prefixes = [
        "cholesky_op",
        "determinant_op",
        "self_adjoint_eig_op",
        "matrix_inverse_op",
        "matrix_solve_ls_op",
        "matrix_solve_op",
        "matrix_triangular_solve_op",
    ],
    deps = [
        ":binary_linalg_ops_common",
        ":linalg_ops_common",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:linalg_ops_op_lib",
        "//third_party/eigen3",
    ],
)

# TODO(josh11b): Should these two *linalg_ops_common libraries be merged?
cc_library(
    name = "linalg_ops_common",
    srcs = ["linalg_ops_common.cc"],
    hdrs = ["linalg_ops_common.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "binary_linalg_ops_common",
    srcs = ["binary_linalg_ops_common.cc"],
    hdrs = ["binary_linalg_ops_common.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

tf_kernel_libraries(
    name = "logging",
    prefixes = [
        "logging_ops",
        "summary_image_op",
        "summary_op",
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
    tests = [
        "logging_ops_test",
        "summary_image_op_test",
        "summary_op_test",
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
        "cast_op",
        "check_numerics_op",
        "cross_op",
        "cwise_op",
        "fft_ops",
        "matmul_op",
        "reduction_ops",
        "segment_reduction_ops",
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

tf_cc_tests(
    tests = [
        "cast_op_test",
        "cross_op_test",
        "cwise_ops_test",
        "matmul_op_test",
        "reduction_ops_test",
        "segment_reduction_ops_test",
        "sparse_matmul_op_test",
    ],
    deps = [
        ":math",
        ":ops_testutil",
        ":ops_util",
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

# conv_grad_ops currently has to be built with conv_ops*.
# TODO(josh11b, zhengxq): put these a separate libraries in ":nn" below once
# conv_ops_gpu.h has be separated into its own library.
tf_kernel_library(
    name = "conv_ops",
    srcs = ["conv_grad_ops.cc"],
    prefix = "conv_ops",
    deps = [
        ":conv_2d",
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
        ":depthwise_conv_op",
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

tf_cc_tests(
    tests = [
        "lrn_op_test",
        "nn_ops_test",
        "xent_op_test",
    ],
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

tf_kernel_library(
    name = "pooling_ops",
    srcs = [
        "avgpooling_op.cc",
        "maxpooling_op.cc",
        "pooling_ops_common.cc",
    ],
    hdrs = [
        "avgpooling_op.h",
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
        ":ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_libraries(
    name = "parsing",
    prefixes = [
        "decode_csv_op",
        "decode_raw_op",
        "example_parsing_ops",
        "string_to_number_op",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:parsing_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "random_ops",
    prefix = "random_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:random_ops_op_lib",
    ],
)

tf_cc_test(
    name = "random_op_test",
    deps = [
        ":random_ops",
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
        "sparse_concat_op",
        "sparse_reorder_op",
        "sparse_split_op",
        "sparse_tensor_dense_matmul_op",
        "sparse_to_dense_op",
        "sparse_xent_op",
        "serialize_sparse_op",
    ],
    deps = [
        ":fill_functor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:sparse_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_cc_tests(
    tests = [
        "sparse_tensor_dense_matmul_op_test",
        "sparse_to_dense_op_test",
        "sparse_xent_op_test",
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
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:string_ops_op_lib",
    ],
)

tf_kernel_library(
    name = "training_ops",
    prefix = "training_ops",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:training_ops_op_lib",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "training_ops_test",
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

# Android libraries -----------------------------------------------------------

filegroup(
    name = "android_srcs",
    srcs = [
        "avgpooling_op.h",
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
        "concat_lib.h",
        "concat_lib_cpu.cc",
        "concat_op.cc",
        "constant_op.cc",
        "constant_op.h",
        "cwise_ops.h",
        "cwise_ops_common.cc",
        "cwise_ops_common.h",
        "dense_update_ops.cc",
        "dense_update_ops.h",
        "example_parsing_ops.cc",
        "fill_functor.h",
        "gather_op.cc",
        "gather_op.h",
        "identity_op.cc",
        "identity_op.h",
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
        "softmax_op.cc",
        "softmax_op.h",
        "split_lib.h",
        "split_lib_cpu.cc",
        "split_op.cc",
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
        "maxpooling_op.h",
        "reduction_ops.h",
        "reduction_ops_common.h",
        "relu_op.h",
        "save_restore_tensor.h",
        "softplus_op.h",
        "softsign_op.h",
        "tile_ops.h",
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
        "conv_ops.cc",
        "cwise_op_add.cc",
        "cwise_op_div.cc",
        "cwise_op_equal_to.cc",
        "cwise_op_exp.cc",
        "cwise_op_greater.cc",
        "cwise_op_isfinite.cc",
        "cwise_op_less.cc",
        "cwise_op_log.cc",
        "cwise_op_maximum.cc",
        "cwise_op_minimum.cc",
        "cwise_op_mul.cc",
        "cwise_op_neg.cc",
        "cwise_op_rsqrt.cc",
        "cwise_op_select.cc",
        "cwise_op_sigmoid.cc",
        "cwise_op_sqrt.cc",
        "cwise_op_square.cc",
        "cwise_op_sub.cc",
        "cwise_op_tanh.cc",
        "dynamic_partition_op.cc",
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
        "reduction_ops_common.cc",
        "reduction_ops_max.cc",
        "reduction_ops_mean.cc",
        "reduction_ops_min.cc",
        "reduction_ops_prod.cc",
        "reduction_ops_sum.cc",
        "relu_op.cc",
        "restore_op.cc",
        "save_op.cc",
        "save_restore_tensor.cc",
        "softplus_op.cc",
        "softsign_op.cc",
        "sparse_to_dense_op.cc",
        "stack_ops.cc",
        "tile_ops.cc",
        "topk_op.cc",
        "training_ops.cc",
        "transpose_functor_cpu.cc",
        "transpose_op.cc",
        "where_op.cc",
        "xent_op.cc",
        ":android_extended_ops_headers",
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
