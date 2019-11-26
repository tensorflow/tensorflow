load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "generate_backend_suites")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "internal",
    includes = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/tf2xla:friends",
    ],
)

generate_backend_suites()

py_library(
    name = "xla_test",
    testonly = 1,
    srcs = ["xla_test.py"],
    srcs_version = "PY2AND3",
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:session",
        "//tensorflow/python:variables",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "xla_test_test",
    size = "small",
    srcs = ["xla_test_test.py"],
    python_version = "PY3",
    deps = [
        ":xla_test",
    ],
)

tf_xla_py_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "adagrad_test",
    size = "small",
    srcs = ["adagrad_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "adagrad_da_test",
    size = "small",
    srcs = ["adagrad_da_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "adam_test",
    size = "small",
    srcs = ["adam_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "add_n_test",
    size = "small",
    srcs = ["add_n_test.py"],
    # TensorList ops are not implemented in the on-demand compilation model yet.
    disabled_backends = ["cpu_ondemand"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "argminmax_test",
    size = "small",
    srcs = ["argminmax_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "binary_ops_test",
    size = "medium",
    srcs = ["binary_ops_test.py"],
    shard_count = 5,
    tags = [
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "complex_div_test",
    size = "medium",
    srcs = ["complex_div_test.py"],
    enabled_backends = [
        "cpu",
        "gpu",
    ],
    shard_count = 2,
    tags = [
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "bucketize_op_test",
    size = "small",
    srcs = ["bucketize_op_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "categorical_op_test",
    size = "small",
    srcs = ["categorical_op_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:standard_ops",
    ],
)

tf_xla_py_test(
    name = "cholesky_op_test",
    size = "medium",
    srcs = ["cholesky_op_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "cond_test",
    size = "small",
    srcs = ["cond_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:function",
    ],
)

tf_xla_py_test(
    name = "self_adjoint_eig_op_test",
    size = "medium",
    srcs = ["self_adjoint_eig_op_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "svd_op_test",
    size = "medium",
    srcs = ["svd_op_test.py"],
    disabled_backends = [
        # TODO(b/129396575): Fails on CPU.
        "cpu",
        "cpu_ondemand",
    ],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "matrix_inverse_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_inverse_op_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "matrix_solve_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_solve_op_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "matrix_triangular_solve_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_triangular_solve_op_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "clustering_test",
    size = "small",
    srcs = ["clustering_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "concat_ops_test",
    size = "medium",
    srcs = ["concat_ops_test.py"],
    tags = ["many_xla_args"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:framework",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "conv2d_test",
    size = "medium",
    srcs = ["conv2d_test.py"],
    shard_count = 10,
    deps = [
        ":test_utils",
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "conv3d_test",
    size = "medium",
    srcs = ["conv3d_test.py"],
    shard_count = 5,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "depthwise_conv_op_test",
    size = "medium",
    srcs = ["depthwise_conv_op_test.py"],
    shard_count = 5,
    tags = [
        "no_rocm",
        "noasan",
        "nomsan",
        "notsan",
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "dynamic_slice_ops_test",
    size = "small",
    srcs = ["dynamic_slice_ops_test.py"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
    ],
)

tf_xla_py_test(
    name = "einsum_op_test",
    size = "medium",
    srcs = ["einsum_op_test.py"],
    enabled_backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:special_math_ops",
    ],
)

tf_xla_py_test(
    name = "reshape_op_test",
    size = "small",
    srcs = ["reshape_op_test.py"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "dynamic_stitch_test",
    size = "small",
    srcs = ["dynamic_stitch_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "extract_image_patches_op_test",
    size = "small",
    srcs = ["extract_image_patches_op_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "eager_test",
    size = "medium",
    srcs = ["eager_test.py"],
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:function",
    ],
)

tf_xla_py_test(
    name = "fifo_queue_test",
    size = "medium",
    srcs = ["fifo_queue_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "fft_test",
    size = "medium",
    srcs = ["fft_test.py"],
    shard_count = 6,
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/signal",
    ],
)

tf_xla_py_test(
    name = "slice_ops_test",
    size = "small",
    srcs = ["slice_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "ftrl_test",
    size = "medium",
    srcs = ["ftrl_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "function_test",
    size = "small",
    srcs = ["function_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "image_ops_test",
    size = "small",
    srcs = ["image_ops_test.py"],
    shard_count = 10,
    tags = [
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "listdiff_op_test",
    size = "small",
    srcs = ["listdiff_op_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform_test",
        "@six_archive//:six",
    ],
)

tf_xla_py_test(
    name = "lrn_ops_test",
    size = "medium",
    srcs = ["lrn_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "manip_ops_test",
    size = "small",
    srcs = ["manip_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:manip_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "matrix_band_part_test",
    size = "medium",
    timeout = "long",
    srcs = ["matrix_band_part_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "matrix_diag_ops_test",
    size = "medium",
    timeout = "long",
    srcs = ["matrix_diag_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "momentum_test",
    size = "small",
    srcs = ["momentum_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "nary_ops_test",
    size = "small",
    srcs = ["nary_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "nullary_ops_test",
    size = "small",
    srcs = ["nullary_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "pooling_ops_test",
    size = "medium",
    srcs = ["pooling_ops_test.py"],
    shard_count = 10,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "pooling_ops_3d_test",
    size = "medium",
    srcs = ["pooling_ops_3d_test.py"],
    shard_count = 10,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "proximal_adagrad_test",
    size = "medium",
    srcs = ["proximal_adagrad_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "proximal_gradient_descent_test",
    size = "medium",
    srcs = ["proximal_gradient_descent_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "qr_op_test",
    size = "medium",
    srcs = ["qr_op_test.py"],
    disabled_backends = [
        # Test is very slow on CPU.
        "cpu",
        "cpu_ondemand",
    ],
    shard_count = 5,
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "random_ops_test",
    size = "small",
    srcs = ["random_ops_test.py"],
    shard_count = 10,
    tags = [
        "notap",  # TODO(b/141057424): flaky on TPU
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:standard_ops",
    ],
)

tf_xla_py_test(
    name = "reduce_ops_test",
    size = "medium",
    srcs = ["reduce_ops_test.py"],
    shard_count = 5,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "reduce_window_test",
    size = "small",
    srcs = ["reduce_window_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "reverse_ops_test",
    size = "medium",
    srcs = ["reverse_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
    ],
)

tf_xla_py_test(
    name = "reverse_sequence_op_test",
    size = "medium",
    srcs = ["reverse_sequence_op_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "rmsprop_test",
    size = "small",
    srcs = ["rmsprop_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "scan_ops_test",
    size = "small",
    srcs = ["scan_ops_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "segment_reduction_ops_test",
    size = "medium",
    srcs = ["segment_reduction_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "spacetobatch_op_test",
    size = "medium",
    srcs = ["spacetobatch_op_test.py"],
    shard_count = 3,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "sparse_to_dense_op_test",
    size = "small",
    srcs = ["sparse_to_dense_op_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_ops",
    ],
)

tf_xla_py_test(
    name = "stack_ops_test",
    size = "small",
    srcs = ["stack_ops_test.py"],
    tags = ["config-cuda-only"],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "stateful_random_ops_test",
    size = "medium",
    srcs = ["stateful_random_ops_test.py"],
    shard_count = 10,
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:stateful_random_ops",
        "//tensorflow/python/kernel_tests/random:util",
    ],
)

tf_xla_py_test(
    name = "stateless_random_ops_test",
    size = "medium",
    srcs = ["stateless_random_ops_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/kernel_tests/random:util",
    ],
)

tf_xla_py_test(
    name = "tensor_array_ops_test",
    size = "medium",
    srcs = ["tensor_array_ops_test.py"],
    # TensorArray ops are not implemented in the on-demand compilation model yet.
    disabled_backends = ["cpu_ondemand"],
    tags = ["config-cuda-only"],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "tensor_list_ops_test",
    size = "small",
    srcs = ["tensor_list_ops_test.py"],
    # TensorList ops are not implemented in the on-demand compilation model yet.
    disabled_backends = ["cpu_ondemand"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:function",
    ],
)

tf_xla_py_test(
    name = "ternary_ops_test",
    size = "medium",
    srcs = ["ternary_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "unary_ops_test",
    size = "medium",
    srcs = ["unary_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "fused_batchnorm_test",
    size = "medium",
    srcs = ["fused_batchnorm_test.py"],
    deps = [
        ":test_utils",
        ":xla_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "variable_ops_test",
    size = "small",
    srcs = ["variable_ops_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "while_test",
    size = "small",
    srcs = ["while_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "gather_test",
    size = "medium",
    srcs = ["gather_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "gather_nd_op_test",
    size = "medium",
    srcs = ["gather_nd_op_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "scatter_nd_op_test",
    size = "medium",
    srcs = ["scatter_nd_op_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "sort_ops_test",
    size = "medium",
    srcs = ["sort_ops_test.py"],
    shard_count = 1,
    # Times out in fastbuild mode.
    tags = ["optonly"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
    ],
)

tf_xla_py_test(
    name = "data_format_ops_test",
    size = "small",
    srcs = ["data_format_ops_test.py"],
    deps = [
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:nn_ops",
    ],
)

tf_xla_py_test(
    name = "xla_device_test",
    size = "small",
    srcs = ["xla_device_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "xla_device_gpu_test",
    size = "small",
    srcs = ["xla_device_gpu_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
    ],
    xla_enable_strict_auto_jit = False,
)

cuda_py_test(
    name = "jit_test",
    size = "medium",
    srcs = ["jit_test.py"],
    additional_deps = [
        ":test_utils",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
    ],
    shard_count = 5,
    xla_enable_strict_auto_jit = False,
)

cuda_py_test(
    name = "dense_layer_test",
    size = "medium",
    srcs = ["dense_layer_test.py"],
    additional_deps = [
        ":test_utils",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:layers",
        "//tensorflow/python:variables",
    ],
    xla_enable_strict_auto_jit = False,
)

cc_library(
    name = "randomized_tests_library",
    testonly = 1,
    srcs = ["randomized_tests.cc"],
    deps = [
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow_opensource",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "randomized_tests",
    size = "large",
    # This test is randomized, so only run it if explicitly requested.
    tags = [
        "manual",
        "notap",
    ] + tf_cuda_tests_tags(),
    deps = [":randomized_tests_library"],
)

tf_cuda_cc_test(
    name = "unary_ops_composition_test",
    srcs = ["unary_ops_composition_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:xla_kernel_creator",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

py_library(
    name = "lstm",
    testonly = 1,
    srcs = ["lstm.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "lstm_test",
    srcs = ["lstm_test.py"],
    additional_deps = [
        ":lstm",
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
    xla_enable_strict_auto_jit = False,
)

# An example of ahead-of-time compilation using tfcompile.  The
# lstm_layer_inference.pbtxt file was generated by running lstm_test
# --dump_graph_dir, and the config file was written by hand.
#
# Run the following to build a minimal benchmark of the computation on Android:
# $ bazel build -c opt --cxxopt='-std=c++11' --linkopt='-lm' \
#   --cpu=armeabi-v7a \
#   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
#   --crosstool_top=//external:android/crosstool \
#   //tensorflow/compiler/tests:lstm_layer_inference_benchmark

#
# Currently the resulting binary size is ~190KB
tf_library(
    name = "lstm_layer_inference",
    testonly = 1,
    config = "lstm_layer_inference.config.pbtxt",
    cpp_class = "LSTMLayerInference",
    graph = "lstm_layer_inference.pbtxt",
    tags = ["manual"],
    tfcompile_flags = ["--xla_cpu_multi_thread_eigen=false"],
)

tf_xla_py_test(
    name = "fake_quant_ops_test",
    size = "medium",
    srcs = ["fake_quant_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "placeholder_test",
    size = "small",
    srcs = ["placeholder_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "quantized_ops_test",
    size = "medium",
    srcs = ["quantized_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "xla_ops_test",
    size = "medium",
    srcs = ["xla_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_test(
    name = "conv_node_name_test",
    size = "medium",
    srcs = ["conv_node_name_test.py"],
    shard_count = 5,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:layers",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "tridiagonal_solve_ops_test",
    size = "medium",
    srcs = ["tridiagonal_solve_ops_test.py"],
    disabled_backends = [
        "cpu_ondemand",
        "cpu",
        "gpu",
    ],
    tags = [
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:framework",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:standard_ops",
    ],
)

tf_xla_py_test(
    name = "determinant_ops_test",
    size = "medium",
    srcs = ["determinant_ops_test.py"],
    disabled_backends = [
        "cpu_ondemand",
        "cpu",
        "gpu",
    ],
    tags = [
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:standard_ops",
    ],
)
