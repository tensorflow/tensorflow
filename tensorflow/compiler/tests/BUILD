licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "generate_backend_suites")

generate_backend_suites()

py_library(
    name = "xla_test",
    testonly = 1,
    srcs = ["xla_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
    ],
)

cc_library(
    name = "depthwise_conv2d_test_kernel",
    testonly = 1,
    srcs = ["depthwise_conv2d_test_kernel.cc"],
    deps = ["//tensorflow/core:framework_lite"],
)

tf_xla_py_test(
    name = "binary_ops_test",
    size = "small",
    srcs = ["binary_ops_test.py"],
    shard_count = 5,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
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
    name = "random_ops_test",
    size = "small",
    srcs = ["random_ops_test.py"],
    # TODO(b/31361304): enable RNG ops on GPU when parallelized.
    disabled_backends = ["gpu"],
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
    size = "small",
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
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cuda_cc_test(
    name = "randomized_tests",
    # This test is randomized, so only run it if explicitly requested.
    tags = [
        "manual",
        "noguitar",
        "notap",
    ],
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
# $ bazel build -c opt --config=android_arm \
#       third_party/tensorflow/compiler/tests:lstm_layer_inference_benchmark
#
# Currently the resulting binary size is ~190KB
tf_library(
    name = "lstm_layer_inference",
    testonly = 1,
    config = "lstm_layer_inference.config.pbtxt",
    cpp_class = "LSTMLayerInference",
    graph = "lstm_layer_inference.pbtxt",
    tags = ["manual"],
    tfcompile_flags = "--xla_cpu_multi_thread_eigen=false",
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
