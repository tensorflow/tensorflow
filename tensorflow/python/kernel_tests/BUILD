# Tests of TensorFlow kernels written using the Python API.

package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_tests")
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

py_library(
    name = "gradient_checker",
    srcs = ["gradient_checker.py"],
    srcs_version = "PY2AND3",
)

medium_kernel_test_list = glob([
    "concat_op_test.py",
    "division_future_test.py",
    "fft_ops_test.py",
    "pooling_ops_test.py",
    "rnn_test.py",
    "random_gamma_test.py",
    "scatter_ops_test.py",
    "seq2seq_test.py",
    "slice_op_test.py",
    "sparse_ops_test.py",
    "sparse_matmul_op_test.py",
    "sparse_tensor_dense_matmul_op_test.py",
])

sharded_kernel_test_list = glob([
    "batch_matrix_band_part_op_test.py",
    "cwise_ops_test.py",
    "embedding_ops_test.py",
    "linalg_grad_test.py",
    "conv_ops_3d_test.py",
])

cpu_only_kernel_test_list = glob([
    "attention_ops_test.py",
    "barrier_ops_test.py",
    "bcast_ops_test.py",
    "benchmark_test.py",
    "candidate_sampler_ops_test.py",
    "cholesky_op_test.py",
    "clip_ops_test.py",
    "decode_csv_op_test.py",
    "decode_raw_op_test.py",
    "determinant_op_test.py",
    "diag_op_test.py",
    "edit_distance_op_test.py",
    "fifo_queue_test.py",
    "identity_op_py_test.py",
    "in_topk_op_test.py",
    "io_ops_test.py",
    "listdiff_op_test.py",
    "logging_ops_test.py",
    "lookup_table_op_test.py",
    "lrn_op_py_test.py",
    "matrix_inverse_op_test.py",
    "matrix_solve_op_test.py",
    "matrix_triangular_solve_op_test.py",
    "matrix_solve_ls_op_test.py",
    "mutex_ops_test.py",
    "parsing_ops_test.py",
    "partitioned_variables_test.py",
    "queue_ops_test.py",
    "random_shuffle_queue_test.py",
    "save_restore_ops_test.py",
    "segment_reduction_ops_test.py",
    "self_adjoint_eig_op_test.py",
    "sparse_add_op_test.py",
    "sparse_concat_op_test.py",
    "sparse_split_op_test.py",
    "sparse_reorder_op_test.py",
    "sparse_reshape_op_test.py",
    "sparse_to_dense_op_test.py",
    "sparsemask_op_test.py",
    "summary_ops_test.py",
    "template_test.py",
    "topk_op_test.py",
    "unique_op_test.py",
    "variable_scope_test.py",
    "variables_test.py",
    "where_op_test.py",
])

py_tests(
    name = "cpu_only_kernel_tests",
    size = "small",
    srcs = cpu_only_kernel_test_list,
)

py_tests(
    name = "reader_ops_test",
    size = "small",
    srcs = ["reader_ops_test.py"],
    additional_deps = [
        "//tensorflow/python:lib",
    ],
)

cuda_py_tests(
    name = "kernel_tests",
    size = "small",
    srcs = glob(
        ["*_test.py"],
        exclude = [
            "reader_ops_test.py",
        ] + cpu_only_kernel_test_list + medium_kernel_test_list + sharded_kernel_test_list,
    ),
)

cuda_py_tests(
    name = "medium_kernel_tests",
    size = "medium",
    srcs = medium_kernel_test_list,
)

cuda_py_tests(
    name = "kernel_tests_with_sharding",
    size = "medium",
    srcs = sharded_kernel_test_list,
    shard_count = 50,
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
