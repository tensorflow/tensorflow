# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

# cuda_py_test and cuda_py_tests enable XLA tests by default. We can't
# combine XLA with TensorRT currently and should set
# xla_enable_strict_auto_jit to False to disable XLA tests.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    glob(["*test.py"]),
)

filegroup(
    name = "tf_trt_integration_test_base_srcs",
    srcs = ["tf_trt_integration_test_base.py"],
    visibility = ["//tensorflow/python/compiler/tensorrt:__pkg__"],
)

filegroup(
    name = "trt_convert_test_data",
    srcs = [
        "testdata/tftrt_2.0_saved_model/saved_model.pb",
        "testdata/tftrt_2.0_saved_model/variables/variables.data-00000-of-00002",
        "testdata/tftrt_2.0_saved_model/variables/variables.data-00001-of-00002",
        "testdata/tftrt_2.0_saved_model/variables/variables.index",
    ],
    visibility = ["//tensorflow/python/compiler/tensorrt:__pkg__"],
)

filegroup(
    name = "quantization_mnist_test_srcs",
    srcs = ["quantization_mnist_test.py"],
    visibility = ["//tensorflow/python/compiler/tensorrt:__pkg__"],
)

filegroup(
    name = "quantization_mnist_test_data",
    srcs = [
        "testdata/mnist/checkpoint",
        "testdata/mnist/model.ckpt-46900.data-00000-of-00001",
        "testdata/mnist/model.ckpt-46900.index",
    ],
    visibility = ["//tensorflow/python/compiler/tensorrt:__pkg__"],
)

cuda_py_tests(
    name = "tf_trt_integration_test",
    srcs = [
        "annotate_max_batch_sizes_test.py",
        "base_test.py",
        "batch_matmul_test.py",
        "biasadd_matmul_test.py",
        "binary_tensor_weight_broadcast_test.py",
        "cast_test.py",
        "combined_nms_test.py",
        "concatenation_test.py",
        "const_broadcast_test.py",
        "conv2d_test.py",
        "dynamic_input_shapes_test.py",
        "identity_output_test.py",
        "int32_test.py",
        "lru_cache_test.py",
        "memory_alignment_test.py",
        "multi_connection_neighbor_engine_test.py",
        "neighboring_engine_test.py",
        "quantization_test.py",
        "rank_two_test.py",
        "reshape_transpose_test.py",
        "tf_function_test.py",
        "topk_test.py",
        "trt_mode_test.py",
        "unary_test.py",
        "vgg_block_nchw_test.py",
        "vgg_block_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # TODO(b/165611343): Need to address the failures for CUDA 11 in OSS build.
        "no_rocm",
        "no_windows",
        "nomac",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/tensorrt:tf_trt_integration_test_base",
    ],
)
