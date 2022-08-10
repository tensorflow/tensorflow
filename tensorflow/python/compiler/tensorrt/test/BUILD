# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

load("//tensorflow/python/compiler/tensorrt:tensorrt.bzl", "tftrt_py_tests")
load("//tensorflow/python/compiler/tensorrt:tensorrt.bzl", "tftrt_py_test")

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

# ============================= TESTS DEFINITION ============================= #

oss_tests = [
    "annotate_max_batch_sizes_test",
    "auto_mixed_precision_test",
    "base_test",
    "batch_matmul_test",
    "biasadd_matmul_test",
    "binary_tensor_weight_broadcast_test",
    "bool_test",
    "cast_test",
    "concatenation_test",
    "const_broadcast_test",
    # "conv2d_test",  # b/198501457
    "data_dependent_shape_test",
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
    "shape_output_test",
    "topk_test",
    "trt_engine_op_shape_test",
    "trt_mode_test",
    "unary_test",
    "vgg_block_nchw_test",
    "vgg_block_test",
]

tftrt_py_tests(
    name = "oss_tests",
    test_names = oss_tests,
)

no_oss_tests = [
    "combined_nms_test",
]

tftrt_py_tests(
    name = "no_oss_tests",
    test_names = no_oss_tests,
    extra_tags = ["no_oss"],
)

# no_oss_tests.append("quantization_mnist_test")
# tftrt_py_test(
#     name = "quantization_mnist_test",
#     srcs = [
#         "//tensorflow/python/compiler/tensorrt/test:quantization_mnist_test_srcs"
#     ],
#     data = [
#         "//tensorflow/python/compiler/tensorrt/test:quantization_mnist_test_data",
#     ],
#     extra_tags = [
#         "no_pip",
#         "no_oss"
#     ],
#     deps = [
#         "//tensorflow/python/compiler/tensorrt:tf_trt_integration_test_base",
#         "//tensorflow/python:client_testlib",
#         "//tensorflow/python:framework_test_lib",
#         "//tensorflow/python/estimator",
#     ],
# )

no_oss_tests.append("tf_function_test")
tftrt_py_test(
    name = "tf_function_test",
    srcs = [
        "tf_function_test.py",
    ],
    extra_tags = [
        "manual",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "no_oss",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "notap",  # TODO(b/231239602): re-enable once naming issue is resolved.
    ],
)

# ========================== TEST SUITES DEFINITION ========================== #

test_suite(
    name = "tf_trt_integration_test",
    tests = oss_tests + no_oss_tests,
)

test_suite(
    name = "tf_trt_integration_test_oss",
    tests = oss_tests,
)

test_suite(
    name = "tf_trt_integration_test_no_oss",
    tests = no_oss_tests,
)
