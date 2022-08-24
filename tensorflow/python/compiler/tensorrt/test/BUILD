# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

# buildifier: disable=same-origin-load
load("//tensorflow/python/compiler/tensorrt:tensorrt.bzl", "tensorrt_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    glob(["*test.py"]),
)

py_library(
    name = "test_utils",
    srcs = [
        "test_utils.py",
    ],
    srcs_version = "PY3",
)

filegroup(
    name = "tf_trt_integration_test_base_srcs",
    srcs = ["tf_trt_integration_test_base.py"],
    visibility = ["//tensorflow/python/compiler/tensorrt:__pkg__"],
)

filegroup(
    name = "trt_convert_test_data",
    srcs = [
        "testdata/tf_readvariableop_saved_model/saved_model.pb",
        "testdata/tf_readvariableop_saved_model/variables/variables.data-00000-of-00001",
        "testdata/tf_readvariableop_saved_model/variables/variables.index",
        "testdata/tf_variablev2_saved_model/saved_model.pb",
        "testdata/tf_variablev2_saved_model/variables/variables.data-00000-of-00001",
        "testdata/tf_variablev2_saved_model/variables/variables.index",
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

oss_tests = [
    "batch_matmul_test",
    "biasadd_matmul_test",
    "binary_tensor_weight_broadcast_test",
    "cast_test",
    "concatenation_test",
    "const_broadcast_test",
    "dynamic_input_shapes_test",
    "identity_output_test",
    "int32_test",
    "lru_cache_test",
    "memory_alignment_test",
    "multi_connection_neighbor_engine_test",
    "neighboring_engine_test",
    "quantization_test",
    "rank_two_test",
    "reshape_transpose_test",
    "topk_test",
    "trt_engine_op_shape_test",
    "trt_mode_test",
    "unary_test",
    "vgg_block_nchw_test",
    "vgg_block_test",
]

no_oss_tests = [
    "base_test",
    "conv2d_test",  # "conv2d_test.py",  # b/198501457
]
# "base_test.py", # TODO(b/165611343): Need to address the failures for CUDA 11 in OSS build.

base_tags = [
    "no_cuda_on_cpu_tap",
    "no_rocm",
    "no_windows",
    "nomac",
]

tensorrt_py_test(
    name = "oss_tests",
    tags = base_tags,
    test_names = oss_tests,
)

tensorrt_py_test(
    name = "no_oss_tests",
    tags = base_tags + ["no_oss"],
    test_names = no_oss_tests + ["combined_nms_test"],
)

tensorrt_py_test(
    name = "tf_function_test",
    tags = base_tags + [
        "manual",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "notap",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "no_oss",  # TODO(b/231239602): re-enable once naming issue is resolved.
    ],
)

test_suite(
    name = "tf_trt_integration_test",
    tests = oss_tests + [
        "combined_nms_test",
        "tf_function_test",
    ],
)

test_suite(
    name = "tf_trt_integration_test_no_oss",
    tests = no_oss_tests,
)
