licenses(["notice"])  # Apache 2.0

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

package(
    default_visibility = [":internal"],
)

load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "generate_backend_suites")
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

generate_backend_suites()

py_library(
    name = "xla_test",
    testonly = 1,
    srcs = ["xla_test.py"],
    srcs_version = "PY2AND3",
    visibility = [":friends"],
    deps = [
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:session",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_test(
    name = "adagrad_test",
    size = "small",
    srcs = ["adagrad_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "argminmax_test",
    size = "small",
    srcs = ["argminmax_test.py"],
    # ArgMax needs CustomCall on CPU, which is not available in normal
    # (not precompiled) TensorFlow. The flag below excludes the CPU
    # backend.
    disabled_backends = "cpu",
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "binary_ops_test",
    size = "small",
    srcs = ["binary_ops_test.py"],
    shard_count = 5,
    tags = [
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "categorical_op_test",
    size = "small",
    srcs = ["categorical_op_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
    ],
)

tf_xla_py_test(
    name = "cholesky_op_test",
    size = "small",
    srcs = ["cholesky_op_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "matrix_triangular_solve_op_test",
    size = "small",
    srcs = ["matrix_triangular_solve_op_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "concat_ops_test",
    size = "medium",
    srcs = ["concat_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:framework_for_generated_wrappers",
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
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "noasan",
        "nomsan",
        "notsan",
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "fft_test",
    size = "medium",
    srcs = ["fft_test.py"],
    shard_count = 3,
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/contrib/signal:signal_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:spectral_ops",
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "ftrl_test",
    size = "small",
    srcs = ["ftrl_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "image_ops_test",
    size = "small",
    srcs = ["image_ops_test.py"],
    tags = [
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "lrn_ops_test",
    size = "medium",
    srcs = ["lrn_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "matrix_band_part_test",
    size = "medium",
    srcs = ["matrix_band_part_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "random_ops_test",
    size = "small",
    srcs = ["random_ops_test.py"],
    # TODO(b/31361304): enable RNG ops on GPU when parallelized.
    disabled_backends = [
        "gpu",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_xla_py_test(
    name = "reverse_sequence_op_test",
    size = "small",
    srcs = ["reverse_sequence_op_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "stack_ops_test",
    size = "small",
    srcs = ["stack_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "stateless_random_ops_test",
    size = "small",
    srcs = ["stateless_random_ops_test.py"],
    tags = ["optonly"],
    deps = [
        ":xla_test",
        "//tensorflow/contrib/stateless",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "tensor_array_ops_test",
    size = "small",
    srcs = ["tensor_array_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
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
    name = "ternary_ops_test",
    size = "small",
    srcs = ["ternary_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "unary_ops_test",
    size = "medium",
    srcs = ["unary_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
    ],
)

tf_xla_py_test(
    name = "fused_batchnorm_test",
    size = "small",
    srcs = ["fused_batchnorm_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "variable_ops_test",
    size = "small",
    srcs = ["variable_ops_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
    ],
)

tf_xla_py_test(
    name = "gather_test",
    size = "medium",
    srcs = ["gather_test.py"],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "xla_device_test",
    size = "small",
    srcs = ["xla_device_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
    ],
)

cuda_py_test(
    name = "jit_test",
    size = "medium",
    srcs = ["jit_test.py"],
    additional_deps = [
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
    ],
    # TODO(b/62961789): Test fails with SIGABRT
    tags = [
        "manual",
        "notap",
    ],
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
        "//tensorflow/core/kernels:ops_util",
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

py_library(
    name = "lstm",
    testonly = 1,
    srcs = ["lstm.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
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
