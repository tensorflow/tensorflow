# Tests of TensorFlow kernels written using the Python API.

package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# CPU only tests should use tf_py_test, GPU tests use cuda_py_test
# Please avoid the py_tests and cuda_py_tests (plural) while we
# fix the shared/overbroad dependencies.

tf_py_test(
    name = "as_string_op_test",
    size = "small",
    srcs = ["as_string_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "attention_ops_test",
    size = "small",
    srcs = ["attention_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "barrier_ops_test",
    size = "small",
    srcs = ["barrier_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "base64_ops_test",
    size = "small",
    srcs = ["base64_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "bcast_ops_test",
    size = "small",
    srcs = ["bcast_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "benchmark_test",
    size = "small",
    srcs = ["benchmark_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "candidate_sampler_ops_test",
    size = "small",
    srcs = ["candidate_sampler_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "cholesky_op_test",
    size = "small",
    srcs = ["cholesky_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "clip_ops_test",
    size = "small",
    srcs = ["clip_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "ctc_decoder_ops_test",
    size = "small",
    srcs = ["ctc_decoder_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "ctc_loss_op_test",
    size = "small",
    srcs = ["ctc_loss_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "decode_csv_op_test",
    size = "small",
    srcs = ["decode_csv_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "decode_png_op_test",
    size = "small",
    srcs = ["decode_png_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "decode_raw_op_test",
    size = "small",
    srcs = ["decode_raw_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "determinant_op_test",
    size = "small",
    srcs = ["determinant_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "edit_distance_op_test",
    size = "small",
    srcs = ["edit_distance_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "fifo_queue_test",
    size = "small",
    srcs = ["fifo_queue_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "fractional_avg_pool_op_test",
    size = "small",
    srcs = ["fractional_avg_pool_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "fractional_max_pool_op_test",
    size = "small",
    srcs = ["fractional_max_pool_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "identity_op_py_test",
    size = "small",
    srcs = ["identity_op_py_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "in_topk_op_test",
    size = "small",
    srcs = ["in_topk_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "io_ops_test",
    size = "small",
    srcs = ["io_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "listdiff_op_test",
    size = "small",
    srcs = ["listdiff_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "logging_ops_test",
    size = "small",
    srcs = ["logging_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "matrix_inverse_op_test",
    size = "small",
    srcs = ["matrix_inverse_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "matrix_solve_ls_op_test",
    size = "small",
    srcs = ["matrix_solve_ls_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "matrix_solve_op_test",
    size = "small",
    srcs = ["matrix_solve_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "matrix_triangular_solve_op_test",
    size = "small",
    srcs = ["matrix_triangular_solve_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "parameterized_truncated_normal_op_test",
    size = "small",
    srcs = ["parameterized_truncated_normal_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "parsing_ops_test",
    size = "small",
    srcs = ["parsing_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "partitioned_variables_test",
    size = "small",
    srcs = ["partitioned_variables_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "priority_queue_test",
    size = "small",
    srcs = ["priority_queue_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "random_shuffle_queue_test",
    size = "small",
    srcs = ["random_shuffle_queue_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "save_restore_ops_test",
    size = "small",
    srcs = ["save_restore_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "segment_reduction_ops_test",
    size = "small",
    srcs = ["segment_reduction_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "sparse_add_op_test",
    size = "small",
    srcs = ["sparse_add_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "sparse_concat_op_test",
    size = "small",
    srcs = ["sparse_concat_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "sparse_reorder_op_test",
    size = "small",
    srcs = ["sparse_reorder_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "sparse_reshape_op_test",
    size = "small",
    srcs = ["sparse_reshape_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "sparse_split_op_test",
    size = "small",
    srcs = ["sparse_split_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "sparse_to_dense_op_py_test",
    size = "small",
    srcs = ["sparse_to_dense_op_py_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "sparsemask_op_test",
    size = "small",
    srcs = ["sparsemask_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "string_join_op_test",
    size = "small",
    srcs = ["string_join_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "string_split_op_test",
    size = "small",
    srcs = ["string_split_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "summary_ops_test",
    size = "small",
    srcs = ["summary_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "summary_tensor_op_test",
    size = "small",
    srcs = ["summary_tensor_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "template_test",
    size = "small",
    srcs = ["template_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "topk_op_test",
    size = "small",
    srcs = ["topk_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "unique_op_test",
    size = "small",
    srcs = ["unique_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "variable_scope_test",
    size = "small",
    srcs = ["variable_scope_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "variables_test",
    size = "small",
    srcs = ["variables_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

tf_py_test(
    name = "where_op_test",
    size = "small",
    srcs = ["where_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

cuda_py_test(
    name = "cast_op_test",
    size = "small",
    srcs = ["cast_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
    tags = ["noasan"],
)

cuda_py_test(
    name = "dense_update_ops_no_tsan_test",
    size = "small",
    srcs = ["dense_update_ops_no_tsan_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
    tags = ["notsan"],
)

tf_py_test(
    name = "diag_op_test",
    size = "medium",
    srcs = ["diag_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
    shard_count = 2,
)

tf_py_test(
    name = "reader_ops_test",
    size = "small",
    srcs = ["reader_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "argmax_op_test",
    size = "small",
    srcs = ["argmax_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "array_ops_test",
    size = "medium",
    srcs = ["array_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "batch_matmul_op_test",
    size = "small",
    srcs = ["batch_matmul_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
    shard_count = 20,
)

cuda_py_test(
    name = "batchtospace_op_test",
    size = "small",
    srcs = ["batchtospace_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "betainc_op_test",
    size = "small",
    srcs = ["betainc_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "bias_op_test",
    size = "small",
    srcs = ["bias_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "bitcast_op_test",
    size = "small",
    srcs = ["bitcast_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "check_ops_test",
    size = "small",
    srcs = ["check_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "control_flow_ops_py_test",
    size = "small",
    srcs = ["control_flow_ops_py_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "conv1d_test",
    size = "small",
    srcs = ["conv1d_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "conv2d_transpose_test",
    size = "small",
    srcs = ["conv2d_transpose_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "conv3d_backprop_filter_v2_grad_test",
    size = "small",
    srcs = ["conv3d_backprop_filter_v2_grad_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "cross_grad_test",
    size = "small",
    srcs = ["cross_grad_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "denormal_test",
    size = "small",
    srcs = ["denormal_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "dense_update_ops_test",
    size = "small",
    srcs = ["dense_update_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "depthtospace_op_test",
    size = "small",
    srcs = ["depthtospace_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "division_past_test",
    size = "medium",
    srcs = ["division_past_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "dynamic_partition_op_test",
    size = "small",
    srcs = ["dynamic_partition_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "dynamic_stitch_op_test",
    size = "small",
    srcs = ["dynamic_stitch_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "extract_image_patches_op_test",
    size = "small",
    srcs = ["extract_image_patches_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "functional_ops_test",
    size = "small",
    srcs = ["functional_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "gather_nd_op_test",
    size = "small",
    srcs = ["gather_nd_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "gather_op_test",
    size = "small",
    srcs = ["gather_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "gradient_correctness_test",
    size = "small",
    srcs = ["gradient_correctness_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "init_ops_test",
    size = "small",
    srcs = ["init_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "linalg_ops_test",
    size = "small",
    srcs = ["linalg_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "lrn_op_test",
    size = "small",
    srcs = ["lrn_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "matmul_op_test",
    size = "small",
    srcs = ["matmul_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "morphological_ops_test",
    size = "small",
    srcs = ["morphological_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "multinomial_op_test",
    size = "small",
    srcs = ["multinomial_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "numerics_test",
    size = "small",
    srcs = ["numerics_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "one_hot_op_test",
    size = "small",
    srcs = ["one_hot_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "pack_op_test",
    size = "small",
    srcs = ["pack_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "pad_op_test",
    size = "small",
    srcs = ["pad_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "padding_fifo_queue_test",
    size = "small",
    srcs = ["padding_fifo_queue_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "py_func_test",
    size = "small",
    srcs = ["py_func_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "random_crop_test",
    size = "small",
    srcs = ["random_crop_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "random_ops_test",
    size = "small",
    srcs = ["random_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "reduce_join_op_test",
    size = "small",
    srcs = ["reduce_join_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "reduction_ops_test",
    size = "medium",
    srcs = ["reduction_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "relu_op_test",
    size = "small",
    srcs = ["relu_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "reshape_op_test",
    size = "small",
    srcs = ["reshape_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "reverse_sequence_op_test",
    size = "small",
    srcs = ["reverse_sequence_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "rnn_cell_test",
    size = "small",
    srcs = ["rnn_cell_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "scalar_strict_test",
    size = "small",
    srcs = ["scalar_strict_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "scan_ops_test",
    size = "medium",
    srcs = ["scan_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "session_ops_test",
    size = "small",
    srcs = ["session_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "shape_ops_test",
    size = "small",
    srcs = ["shape_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "softmax_op_test",
    size = "small",
    srcs = ["softmax_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "softplus_op_test",
    size = "small",
    srcs = ["softplus_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "softsign_op_test",
    size = "small",
    srcs = ["softsign_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "spacetobatch_op_test",
    size = "small",
    srcs = ["spacetobatch_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "spacetodepth_op_test",
    size = "small",
    srcs = ["spacetodepth_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "sparse_serialization_ops_test",
    size = "small",
    srcs = ["sparse_serialization_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "sparse_tensor_dense_matmul_grad_test",
    size = "small",
    srcs = ["sparse_tensor_dense_matmul_grad_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "sparse_xent_op_test",
    size = "small",
    srcs = ["sparse_xent_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "split_op_test",
    size = "small",
    srcs = ["split_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "stack_ops_test",
    size = "small",
    srcs = ["stack_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "string_to_hash_bucket_op_test",
    size = "small",
    srcs = ["string_to_hash_bucket_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "string_to_number_op_test",
    size = "small",
    srcs = ["string_to_number_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "summary_audio_op_test",
    size = "small",
    srcs = ["summary_audio_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "summary_image_op_test",
    size = "small",
    srcs = ["summary_image_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "tensor_array_ops_test",
    size = "small",
    srcs = ["tensor_array_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "trace_op_test",
    size = "small",
    srcs = ["trace_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "transpose_op_test",
    size = "small",
    srcs = ["transpose_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "unpack_op_test",
    size = "small",
    srcs = ["unpack_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "variable_ops_test",
    size = "small",
    srcs = ["variable_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "xent_op_test",
    size = "small",
    srcs = ["xent_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "zero_division_test",
    size = "small",
    srcs = ["zero_division_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "atrous_conv2d_test",
    size = "medium",
    srcs = ["atrous_conv2d_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "atrous_convolution_test",
    size = "medium",
    srcs = ["atrous_convolution_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "pool_test",
    size = "medium",
    srcs = ["pool_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "conv2d_backprop_filter_grad_test",
    size = "medium",
    srcs = ["conv2d_backprop_filter_grad_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "conv3d_transpose_test",
    size = "medium",
    srcs = ["conv3d_transpose_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "conv_ops_test",
    size = "medium",
    srcs = ["conv_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "depthwise_conv_op_test",
    size = "medium",  # http://b/30603882
    srcs = ["depthwise_conv_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "division_future_test",
    size = "medium",
    srcs = ["division_future_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "fft_ops_test",
    size = "medium",
    srcs = ["fft_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "pooling_ops_3d_test",
    size = "medium",  # http://b/30600785
    srcs = ["pooling_ops_3d_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "pooling_ops_test",
    size = "medium",
    srcs = ["pooling_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "random_gamma_test",
    size = "medium",
    srcs = ["random_gamma_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "rnn_test",
    size = "medium",
    srcs = ["rnn_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "scatter_ops_test",
    size = "large",  # NOTE: This is not run.
    srcs = ["scatter_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "seq2seq_test",
    size = "medium",
    srcs = ["seq2seq_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "slice_op_test",
    size = "medium",
    srcs = ["slice_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "sparse_matmul_op_test",
    size = "medium",
    srcs = ["sparse_matmul_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "sparse_ops_test",
    size = "medium",
    srcs = ["sparse_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "sparse_tensor_dense_matmul_op_test",
    size = "medium",
    srcs = ["sparse_tensor_dense_matmul_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
)

# TODO(gpapan): Revisit the gradient of extract_image_patches_op to resolve
# http://b/31080670.
cuda_py_test(
    name = "extract_image_patches_grad_test",
    size = "medium",
    srcs = ["extract_image_patches_grad_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
    tags = ["notap"],  # http://b/31080670
)

cuda_py_test(
    name = "concat_op_test",
    size = "medium",
    srcs = ["concat_op_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
    ],
    tags = ["notsan"],  # http://b/30445083
)

cuda_py_test(
    name = "conv_ops_3d_test",
    size = "medium",
    srcs = ["conv_ops_3d_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    shard_count = 20,
)

cuda_py_test(
    name = "cwise_ops_test",
    size = "medium",
    srcs = ["cwise_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    shard_count = 50,
    tags = ["notap"],  # b/30226163
)

cuda_py_test(
    name = "embedding_ops_test",
    size = "medium",
    srcs = ["embedding_ops_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    shard_count = 20,
)

cuda_py_test(
    name = "linalg_grad_test",
    size = "medium",
    srcs = ["linalg_grad_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    shard_count = 20,
)

cuda_py_test(
    name = "matrix_band_part_op_test",
    size = "medium",
    srcs = ["matrix_band_part_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    shard_count = 20,
)

cuda_py_test(
    name = "self_adjoint_eig_op_test",
    size = "medium",
    srcs = ["self_adjoint_eig_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    shard_count = 20,
)

cuda_py_test(
    name = "svd_op_test",
    size = "medium",
    srcs = ["svd_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    shard_count = 20,
)

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
