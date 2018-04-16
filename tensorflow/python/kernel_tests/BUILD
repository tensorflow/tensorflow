# Tests of TensorFlow kernels written using the Python API.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "sycl_py_test")

# CPU only tests should use tf_py_test, GPU tests use cuda_py_test
# Please avoid the py_tests and cuda_py_tests (plural) while we
# fix the shared/overbroad dependencies.

tf_py_test(
    name = "as_string_op_test",
    size = "small",
    srcs = ["as_string_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
    ],
    tags = ["no_windows"],
)

tf_py_test(
    name = "attention_ops_test",
    size = "small",
    srcs = ["attention_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
    ],
)

tf_py_test(
    name = "barrier_ops_test",
    size = "medium",  # NOTE(ebrevdo): This test is NOT small.
    srcs = ["barrier_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    shard_count = 20,
)

tf_py_test(
    name = "base64_ops_test",
    size = "small",
    srcs = ["base64_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:string_ops",
    ],
    tags = ["nomac"],  # b/35468214
)

tf_py_test(
    name = "bcast_ops_test",
    size = "small",
    srcs = ["bcast_ops_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "list_ops_test",
    size = "small",
    srcs = ["list_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:client_testlib",
    ],
    grpc_enabled = True,
)

cuda_py_test(
    name = "benchmark_test",
    size = "small",
    srcs = ["benchmark_test.py"],
    additional_deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_benchmark",
    ],
    tags = ["no_windows"],
)

tf_py_test(
    name = "bincount_op_test",
    size = "small",
    srcs = ["bincount_op_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "candidate_sampler_ops_test",
    size = "small",
    srcs = ["candidate_sampler_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:candidate_sampling_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

tf_py_test(
    name = "checkpoint_ops_test",
    size = "medium",
    srcs = ["checkpoint_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:checkpoint_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "cholesky_op_test",
    size = "medium",
    srcs = ["cholesky_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/ops/linalg",
    ],
    shard_count = 5,
    tags = ["no_windows_gpu"],
)

tf_py_test(
    name = "clip_ops_test",
    size = "small",
    srcs = ["clip_ops_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    tags = ["no_windows"],
)

tf_py_test(
    name = "conditional_accumulator_test",
    size = "small",
    srcs = ["conditional_accumulator_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "ctc_decoder_ops_test",
    size = "small",
    srcs = ["ctc_decoder_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:ctc_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "ctc_loss_op_test",
    size = "small",
    srcs = ["ctc_loss_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:ctc_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
    ],
)

tf_py_test(
    name = "decode_csv_op_test",
    size = "small",
    srcs = ["decode_csv_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:parsing_ops",
    ],
)

tf_py_test(
    name = "decode_png_op_test",
    size = "small",
    srcs = ["decode_png_op_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:nn_grad",
    ],
)

tf_py_test(
    name = "decode_bmp_op_test",
    size = "small",
    srcs = ["decode_bmp_op_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:nn_grad",
    ],
)

tf_py_test(
    name = "decode_jpeg_op_test",
    srcs = ["decode_jpeg_op_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:platform",
    ],
    data = ["//tensorflow/core:image_testdata"],
)

tf_py_test(
    name = "decode_image_op_test",
    size = "small",
    srcs = ["decode_image_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:nn_grad",
    ],
    data = ["//tensorflow/core:image_testdata"],
)

tf_py_test(
    name = "decode_raw_op_test",
    size = "small",
    srcs = ["decode_raw_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
    ],
)

tf_py_test(
    name = "decode_compressed_op_test",
    size = "small",
    srcs = ["decode_compressed_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
    ],
)

cuda_py_test(
    name = "determinant_op_test",
    size = "small",
    srcs = ["determinant_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
    ],
)

tf_py_test(
    name = "draw_bounding_box_op_test",
    size = "small",
    srcs = ["draw_bounding_box_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
    ],
)

tf_py_test(
    name = "edit_distance_op_test",
    size = "small",
    srcs = ["edit_distance_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "fifo_queue_test",
    size = "small",
    srcs = ["fifo_queue_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "fractional_avg_pool_op_test",
    size = "small",
    srcs = ["fractional_avg_pool_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
    ],
    shard_count = 5,
)

tf_py_test(
    name = "fractional_max_pool_op_test",
    size = "small",
    srcs = ["fractional_max_pool_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
    ],
    shard_count = 5,
)

tf_py_test(
    name = "identity_op_py_test",
    size = "small",
    srcs = ["identity_op_py_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "identity_n_op_py_test",
    size = "small",
    srcs = ["identity_n_op_py_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "in_topk_op_test",
    size = "small",
    srcs = ["in_topk_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:nn_ops",
    ],
)

tf_py_test(
    name = "record_input_test",
    size = "medium",
    srcs = ["record_input_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "io_ops_test",
    size = "small",
    srcs = ["io_ops_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "listdiff_op_test",
    size = "small",
    srcs = ["listdiff_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "logging_ops_test",
    size = "small",
    srcs = ["logging_ops_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
    ],
)

tf_py_test(
    name = "lookup_ops_test",
    size = "small",
    srcs = ["lookup_ops_test.py"],
    additional_deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
    ],
    grpc_enabled = True,
)

tf_py_test(
    name = "losses_test",
    size = "medium",
    srcs = ["losses_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "matrix_exponential_op_test",
    size = "small",
    srcs = ["matrix_exponential_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
    ],
)

tf_py_test(
    name = "matrix_logarithm_op_test",
    size = "small",
    srcs = ["matrix_logarithm_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
    ],
)

cuda_py_test(
    name = "matrix_inverse_op_test",
    size = "small",
    srcs = ["matrix_inverse_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "matrix_solve_ls_op_test",
    size = "medium",
    srcs = ["matrix_solve_ls_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "matrix_solve_op_test",
    size = "small",
    srcs = ["matrix_solve_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
    ],
)

cuda_py_test(
    name = "matrix_triangular_solve_op_test",
    size = "small",
    srcs = ["matrix_triangular_solve_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:linalg_ops",
    ],
)

cuda_py_test(
    name = "parameterized_truncated_normal_op_test",
    size = "small",
    srcs = ["parameterized_truncated_normal_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
    ],
)

tf_py_test(
    name = "parsing_ops_test",
    size = "small",
    srcs = ["parsing_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "parse_single_example_op_test",
    size = "small",
    srcs = ["parse_single_example_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "partitioned_variables_test",
    size = "small",
    srcs = ["partitioned_variables_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "priority_queue_test",
    size = "small",
    srcs = ["priority_queue_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
    ],
)

cuda_py_test(
    name = "resource_variable_ops_test",
    size = "small",
    srcs = ["resource_variable_ops_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "regex_replace_op_test",
    size = "small",
    srcs = ["regex_replace_op_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:string_ops",
    ],
)

tf_py_test(
    name = "save_restore_ops_test",
    size = "small",
    srcs = ["save_restore_ops_test.py"],
    additional_deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:io_ops_gen",
    ],
)

cuda_py_test(
    name = "scatter_nd_ops_test",
    size = "medium",
    srcs = ["scatter_nd_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python:resource_variable_ops",
    ],
    tags = ["noasan"],  # http://b/32635055
)

tf_py_test(
    name = "segment_reduction_ops_test",
    size = "medium",
    srcs = ["segment_reduction_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python:nn_grad",
    ],
)

tf_py_test(
    name = "sparse_add_op_test",
    size = "small",
    srcs = ["sparse_add_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "sparse_concat_op_test",
    size = "small",
    srcs = ["sparse_concat_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "sparse_conditional_accumulator_test",
    size = "small",
    srcs = ["sparse_conditional_accumulator_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "sparse_reorder_op_test",
    size = "small",
    srcs = ["sparse_reorder_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "sparse_reshape_op_test",
    size = "small",
    srcs = ["sparse_reshape_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "sparse_split_op_test",
    size = "small",
    srcs = ["sparse_split_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "sparse_slice_op_test",
    size = "small",
    srcs = ["sparse_slice_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "sparse_to_dense_op_py_test",
    size = "small",
    srcs = ["sparse_to_dense_op_py_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "sparsemask_op_test",
    size = "small",
    srcs = ["sparsemask_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "string_join_op_test",
    size = "small",
    srcs = ["string_join_op_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:string_ops",
    ],
)

tf_py_test(
    name = "string_split_op_test",
    size = "small",
    srcs = ["string_split_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
    ],
)

tf_py_test(
    name = "substr_op_test",
    size = "small",
    srcs = ["substr_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:string_ops",
    ],
)

tf_py_test(
    name = "summary_ops_test",
    size = "small",
    srcs = ["summary_ops_test.py"],
    additional_deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:summary",
    ],
)

tf_py_test(
    name = "summary_tensor_op_test",
    size = "small",
    srcs = ["summary_tensor_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:summary_ops",
    ],
)

tf_py_test(
    name = "template_test",
    size = "small",
    srcs = ["template_test.py"],
    additional_deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:template",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "topk_op_test",
    size = "small",
    srcs = ["topk_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "nth_element_op_test",
    size = "small",
    srcs = ["nth_element_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

tf_py_test(
    name = "unique_op_test",
    size = "small",
    srcs = ["unique_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "variable_scope_test",
    size = "small",
    srcs = ["variable_scope_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
    ],
    tags = ["no_windows"],
)

tf_py_test(
    name = "variables_test",
    size = "small",
    srcs = ["variables_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:function",
    ],
)

cuda_py_test(
    name = "where_op_test",
    size = "medium",
    srcs = ["where_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "cast_op_test",
    size = "small",
    srcs = ["cast_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
    ],
    tags = [
        "no_windows",
        "noasan",
        "noguitar",
        "notap",
    ],
)

cuda_py_test(
    name = "dense_update_ops_no_tsan_test",
    size = "small",
    srcs = ["dense_update_ops_no_tsan_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
    tags = ["notsan"],
)

cuda_py_test(
    name = "diag_op_test",
    size = "medium",
    srcs = ["diag_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:platform",
    ],
    shard_count = 2,
    tags = ["no_windows_gpu"],
)

tf_py_test(
    name = "reader_ops_test",
    size = "small",
    srcs = ["reader_ops_test.py"],
    additional_deps = [
        "@six_archive//:six",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
    data = ["//tensorflow/core:lmdb_testdata"],
)

cuda_py_test(
    name = "aggregate_ops_test",
    size = "small",
    srcs = ["aggregate_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "argmax_op_test",
    size = "small",
    srcs = ["argmax_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "array_ops_test",
    size = "medium",
    srcs = ["array_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:test_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
    ],
    shard_count = 10,
)

cuda_py_test(
    name = "broadcast_to_ops_test",
    size = "small",
    srcs = ["broadcast_to_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "inplace_ops_test",
    size = "small",
    srcs = ["inplace_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
    ],
    shard_count = 10,
)

cuda_py_test(
    name = "batch_matmul_op_test",
    size = "small",
    srcs = ["batch_matmul_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
    shard_count = 20,
)

cuda_py_test(
    name = "batchtospace_op_test",
    size = "small",
    srcs = ["batchtospace_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "betainc_op_test",
    size = "small",
    srcs = ["betainc_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

cuda_py_test(
    name = "bias_op_test",
    size = "small",
    srcs = ["bias_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "bitcast_op_test",
    size = "small",
    srcs = ["bitcast_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "check_ops_test",
    size = "small",
    srcs = ["check_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "constant_op_eager_test",
    size = "small",
    srcs = ["constant_op_eager_test.py"],
    additional_deps = [
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "control_flow_ops_py_test",
    # TODO(b/70473603): change this back to "small" once the C API is
    # permanently enabled
    size = "medium",
    srcs = ["control_flow_ops_py_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:data_flow_ops_gen",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:logging_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "control_flow_util_test",
    size = "small",
    srcs = ["control_flow_util_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_ops_gen",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:test_ops",
    ],
)

cuda_py_test(
    name = "conv1d_test",
    size = "small",
    srcs = ["conv1d_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "conv2d_transpose_test",
    size = "small",
    srcs = ["conv2d_transpose_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "conv3d_backprop_filter_v2_grad_test",
    size = "small",
    srcs = ["conv3d_backprop_filter_v2_grad_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "cross_grad_test",
    size = "small",
    srcs = ["cross_grad_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "denormal_test",
    size = "small",
    srcs = ["denormal_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
    ],
)

cuda_py_test(
    name = "dense_update_ops_test",
    size = "small",
    srcs = ["dense_update_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "depthtospace_op_test",
    size = "medium",
    srcs = ["depthtospace_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "division_past_test",
    size = "medium",
    srcs = ["division_past_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    tags = ["manual"],
)

cuda_py_test(
    name = "dynamic_partition_op_test",
    size = "medium",
    srcs = ["dynamic_partition_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_grad",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
    ],
)

cuda_py_test(
    name = "dynamic_stitch_op_test",
    size = "small",
    srcs = ["dynamic_stitch_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_grad",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
    ],
)

cuda_py_test(
    name = "extract_image_patches_op_test",
    size = "small",
    srcs = ["extract_image_patches_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "functional_ops_test",
    size = "small",
    srcs = ["functional_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    grpc_enabled = True,
    tags = ["no_windows"],
)

cuda_py_test(
    name = "gather_nd_op_test",
    size = "small",
    srcs = ["gather_nd_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "gather_op_test",
    size = "medium",
    srcs = ["gather_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
    ],
)

cuda_py_test(
    name = "gradient_correctness_test",
    size = "small",
    srcs = ["gradient_correctness_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "init_ops_test",
    size = "medium",
    srcs = ["init_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:layers",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "linalg_ops_test",
    size = "medium",
    srcs = ["linalg_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/ops/linalg",
    ],
    tags = ["no_windows_gpu"],
)

cuda_py_test(
    name = "lrn_op_test",
    size = "small",
    srcs = ["lrn_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_grad",
    ],
)

cuda_py_test(
    name = "manip_ops_test",
    size = "small",
    srcs = ["manip_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:manip_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    tags = ["no_windows_gpu"],
)

cuda_py_test(
    name = "matmul_op_test",
    size = "small",
    srcs = ["matmul_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
    ],
    shard_count = 20,
)

cuda_py_test(
    name = "morphological_ops_test",
    size = "small",
    srcs = ["morphological_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "numerics_test",
    size = "small",
    srcs = ["numerics_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:numerics",
    ],
)

cuda_py_test(
    name = "one_hot_op_test",
    size = "small",
    srcs = ["one_hot_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    tags = ["no_windows_gpu"],
)

cuda_py_test(
    name = "stack_op_test",
    size = "small",
    srcs = ["stack_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "pad_op_test",
    size = "small",
    srcs = ["pad_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "padding_fifo_queue_test",
    size = "small",
    srcs = ["padding_fifo_queue_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "py_func_test",
    size = "small",
    srcs = ["py_func_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:script_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
    ],
    tags = ["no_windows"],
)

cuda_py_test(
    name = "reduce_join_op_test",
    size = "small",
    srcs = ["reduce_join_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
    ],
)

cuda_py_test(
    name = "reduction_ops_test",
    size = "medium",
    srcs = ["reduction_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
    shard_count = 4,
    tags = ["no_windows_gpu"],
)

cuda_py_test(
    name = "reduction_ops_test_big",
    size = "medium",
    srcs = ["reduction_ops_test_big.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
    tags = [
        "manual",
        "no_gpu",
        "nogpu",
        "noguitar",
        "notap",
    ],
)

cuda_py_test(
    name = "relu_op_test",
    size = "small",
    srcs = ["relu_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "reshape_op_test",
    size = "small",
    srcs = ["reshape_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "reverse_sequence_op_test",
    size = "small",
    srcs = ["reverse_sequence_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "compare_and_bitpack_op_test",
    size = "small",
    srcs = ["compare_and_bitpack_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "scalar_test",
    size = "small",
    srcs = ["scalar_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
    ],
)

cuda_py_test(
    name = "scan_ops_test",
    size = "medium",
    srcs = ["scan_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "session_ops_test",
    size = "small",
    srcs = ["session_ops_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session_ops",
    ],
)

cuda_py_test(
    name = "shape_ops_test",
    size = "medium",
    srcs = ["shape_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
    ],
)

cuda_py_test(
    name = "softmax_op_test",
    size = "medium",
    srcs = ["softmax_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "softplus_op_test",
    size = "small",
    srcs = ["softplus_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "softsign_op_test",
    size = "small",
    srcs = ["softsign_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "spacetobatch_op_test",
    size = "small",
    srcs = ["spacetobatch_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "spacetodepth_op_test",
    size = "medium",
    srcs = ["spacetodepth_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

tf_py_test(
    name = "sparse_serialization_ops_test",
    size = "small",
    srcs = ["sparse_serialization_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "sparse_tensors_map_ops_test",
    size = "small",
    srcs = ["sparse_tensors_map_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "sparse_tensor_dense_matmul_grad_test",
    size = "small",
    srcs = ["sparse_tensor_dense_matmul_grad_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:sparse_ops",
    ],
)

cuda_py_test(
    name = "sparse_xent_op_test",
    size = "small",
    srcs = ["sparse_xent_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "split_op_test",
    size = "medium",
    srcs = ["split_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "stack_ops_test",
    size = "small",
    srcs = ["stack_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops_gen",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "string_to_hash_bucket_op_test",
    size = "small",
    srcs = ["string_to_hash_bucket_op_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:string_ops",
    ],
)

cuda_py_test(
    name = "string_to_number_op_test",
    size = "small",
    srcs = ["string_to_number_op_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
    ],
    tags = ["no_windows"],
)

cuda_py_test(
    name = "summary_audio_op_test",
    size = "small",
    srcs = ["summary_audio_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:summary",
    ],
)

cuda_py_test(
    name = "summary_image_op_test",
    size = "small",
    srcs = ["summary_image_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:summary",
    ],
)

cuda_py_test(
    name = "tensor_array_ops_test",
    size = "small",
    srcs = ["tensor_array_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops_gen",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:training",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
    ],
    flaky = 1,  # create_local_cluster sometimes times out.
)

cuda_py_test(
    name = "trace_op_test",
    size = "small",
    srcs = ["trace_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
    ],
    tags = ["no_windows_gpu"],
)

cuda_py_test(
    name = "transpose_op_test",
    size = "large",
    srcs = ["transpose_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    shard_count = 2,
    tags = [
        "no_gpu",
        "no_oss",
    ],
)

cuda_py_test(
    name = "unstack_op_test",
    size = "small",
    srcs = ["unstack_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "variable_ops_test",
    size = "small",
    srcs = ["variable_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "xent_op_test",
    size = "small",
    srcs = ["xent_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
    ],
)

cuda_py_test(
    name = "zero_division_test",
    size = "small",
    srcs = ["zero_division_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "atrous_conv2d_test",
    size = "medium",
    srcs = ["atrous_conv2d_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
    shard_count = 2,
)

cuda_py_test(
    name = "atrous_convolution_test",
    size = "medium",
    srcs = ["atrous_convolution_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
    tags = ["manual"],
)

cuda_py_test(
    name = "pool_test",
    size = "medium",
    srcs = ["pool_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "conv2d_backprop_filter_grad_test",
    size = "medium",
    srcs = ["conv2d_backprop_filter_grad_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "conv3d_transpose_test",
    size = "medium",
    srcs = ["conv3d_transpose_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "conv_ops_test",
    size = "large",
    srcs = ["conv_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
    ],
    shard_count = 4,
)

cuda_py_test(
    name = "depthwise_conv_op_test",
    size = "medium",  # http://b/30603882
    srcs = ["depthwise_conv_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

tf_py_test(
    name = "neon_depthwise_conv_op_test",
    size = "medium",
    srcs = ["neon_depthwise_conv_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
    tags = ["no_windows"],
)

cuda_py_test(
    name = "division_future_test",
    size = "medium",
    srcs = ["division_future_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    tags = ["manual"],
)

cuda_py_test(
    name = "dct_ops_test",
    srcs = ["dct_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:spectral_ops",
        "//tensorflow/python:spectral_ops_test_util",
    ],
)

cuda_py_test(
    name = "fft_ops_test",
    size = "large",
    srcs = ["fft_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:spectral_ops",
        "//tensorflow/python:spectral_ops_test_util",
    ],
    shard_count = 3,
)

cuda_py_test(
    name = "pooling_ops_3d_test",
    size = "medium",
    srcs = ["pooling_ops_3d_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "pooling_ops_test",
    size = "medium",
    srcs = ["pooling_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
    ],
    shard_count = 4,
)

cuda_py_test(
    name = "rnn_test",
    size = "medium",
    srcs = ["rnn_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_grad",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
    ],
    shard_count = 10,
)

cuda_py_test(
    name = "scatter_ops_test",
    size = "large",  # NOTE: This is not run by default.
    srcs = ["scatter_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "slice_op_test",
    size = "large",
    srcs = ["slice_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
    ],
)

cuda_py_test(
    name = "sparse_matmul_op_test",
    size = "medium",
    srcs = ["sparse_matmul_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
    tags = ["no_windows"],
)

cuda_py_test(
    name = "sparse_ops_test",
    size = "medium",
    srcs = ["sparse_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_grad",
        "//tensorflow/python:sparse_ops",
    ],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",  # b/77589990
    ],
)

cuda_py_test(
    name = "sparse_tensor_dense_matmul_op_test",
    size = "medium",
    srcs = ["sparse_tensor_dense_matmul_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
    ],
)

# TODO(gpapan): Revisit the gradient of extract_image_patches_op to resolve
# http://b/31080670.
cuda_py_test(
    name = "extract_image_patches_grad_test",
    size = "medium",
    srcs = ["extract_image_patches_grad_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    tags = ["notap"],  # http://b/31080670
)

cuda_py_test(
    name = "stage_op_test",
    size = "medium",
    srcs = ["stage_op_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:data_flow_ops",
    ],
)

cuda_py_test(
    name = "map_stage_op_test",
    size = "medium",
    srcs = ["map_stage_op_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:data_flow_ops",
    ],
)

cuda_py_test(
    name = "concat_op_test",
    size = "medium",
    srcs = ["concat_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "large_concat_op_test",
    size = "medium",
    srcs = ["large_concat_op_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    tags = [
        "nomsan",
        "notsan",
    ],
)

cuda_py_test(
    name = "conv_ops_3d_test",
    size = "medium",
    srcs = ["conv_ops_3d_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
    ],
    shard_count = 20,
)

cuda_py_test(
    name = "cwise_ops_test",
    size = "medium",
    srcs = ["cwise_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
    shard_count = 50,
)

cuda_py_test(
    name = "embedding_ops_test",
    size = "medium",
    srcs = ["embedding_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    shard_count = 20,
)

cuda_py_test(
    name = "linalg_grad_test",
    size = "medium",
    srcs = ["linalg_grad_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/ops/linalg",
    ],
    shard_count = 20,
)

cuda_py_test(
    name = "matrix_band_part_op_test",
    size = "medium",
    srcs = ["matrix_band_part_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
    shard_count = 20,
)

cuda_py_test(
    name = "self_adjoint_eig_op_test",
    size = "medium",
    srcs = ["self_adjoint_eig_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
    ],
    data = ["//tensorflow/python/kernel_tests/testdata:self_adjoint_eig_op_test_files"],
    shard_count = 20,
    tags = ["no_windows"],
)

cuda_py_test(
    name = "qr_op_test",
    size = "medium",
    srcs = ["qr_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
    ],
    shard_count = 20,
)

cuda_py_test(
    name = "svd_op_test",
    size = "medium",
    srcs = ["svd_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
    ],
    shard_count = 20,
)

cuda_py_test(
    name = "norm_op_test",
    size = "medium",
    srcs = ["norm_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
    ],
    shard_count = 20,
    tags = ["no_windows_gpu"],
)

cuda_py_test(
    name = "tensordot_op_test",
    size = "medium",
    srcs = ["tensordot_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
    shard_count = 20,
)

sycl_py_test(
    name = "basic_gpu_test",
    size = "small",
    srcs = ["basic_gpu_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
    ],
)

tf_py_test(
    name = "sets_test",
    size = "medium",
    srcs = ["sets_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sets",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "weights_broadcast_test",
    size = "small",
    srcs = ["weights_broadcast_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:weights_broadcast_ops",
    ],
    shard_count = 3,
)

tf_py_test(
    name = "metrics_test",
    size = "medium",
    srcs = ["metrics_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_grad",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
    ],
    shard_count = 10,
    tags = ["no_windows_gpu"],
)

tf_py_test(
    name = "confusion_matrix_test",
    size = "small",
    srcs = ["confusion_matrix_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:confusion_matrix",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
    ],
)

tf_py_test(
    name = "bucketize_op_test",
    size = "small",
    srcs = ["bucketize_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

tf_py_test(
    name = "sparse_cross_op_test",
    size = "small",
    srcs = ["sparse_cross_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_py_test(
    name = "garbage_collection_test",
    size = "small",
    srcs = ["garbage_collection_test.py"],
    additional_deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "accumulate_n_test",
    size = "small",
    srcs = ["accumulate_n_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "accumulate_n_eager_test",
    size = "small",
    srcs = ["accumulate_n_eager_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
    ],
)
