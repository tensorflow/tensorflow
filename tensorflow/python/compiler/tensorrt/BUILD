# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

# cuda_py_test and cuda_py_tests enable XLA tests by default. We can't
# combine XLA with TensorRT currently and should set
# xla_enable_strict_auto_jit to False to disable XLA tests.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

exports_files(glob([
    "test/testdata/*",
]))

py_library(
    name = "init_py",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":tf_trt_integration_test_base",
        ":trt_convert_py",
    ],
)

py_library(
    name = "trt_convert_py",
    srcs = [
        "trt_convert.py",
        "utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/tf2tensorrt:_pywrap_py_utils",
        "//tensorflow/compiler/tf2tensorrt:trt_engine_instance_proto_py",
        "//tensorflow/compiler/tf2tensorrt:trt_ops_loader",
        "//tensorflow/python:convert_to_constants",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:graph_util",
        "//tensorflow/python:session",
        "//tensorflow/python:tf_optimizer",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

py_library(
    name = "trt_convert_windows",
    srcs = ["trt_convert_windows.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "tf_trt_integration_test_base",
    srcs = ["test/tf_trt_integration_test_base.py"],
    srcs_version = "PY3",
    deps = [
        ":trt_convert_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "//tensorflow/python/tools:saved_model_utils",
    ],
)

cuda_py_test(
    name = "trt_convert_test",
    srcs = ["trt_convert_test.py"],
    data = [
        "test/testdata/tftrt_2.0_saved_model/saved_model.pb",
        "test/testdata/tftrt_2.0_saved_model/variables/variables.data-00000-of-00002",
        "test/testdata/tftrt_2.0_saved_model/variables/variables.data-00001-of-00002",
        "test/testdata/tftrt_2.0_saved_model/variables/variables.index",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_pip",
        "no_windows",
        "nomac",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":trt_convert_py",
        "//tensorflow/compiler/tf2tensorrt:trt_engine_instance_proto_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:graph_util",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "//tensorflow/python/tools:freeze_graph_lib",
        "//tensorflow/python/tools:saved_model_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_tests(
    name = "tf_trt_integration_test",
    srcs = [
        "test/annotate_max_batch_sizes_test.py",
        "test/base_test.py",
        "test/batch_matmul_test.py",
        "test/biasadd_matmul_test.py",
        "test/binary_tensor_weight_broadcast_test.py",
        "test/cast_test.py",
        "test/combined_nms_test.py",
        "test/concatenation_test.py",
        "test/const_broadcast_test.py",
        "test/conv2d_test.py",
        "test/dynamic_input_shapes_test.py",
        "test/identity_output_test.py",
        "test/int32_test.py",
        "test/lru_cache_test.py",
        "test/memory_alignment_test.py",
        "test/multi_connection_neighbor_engine_test.py",
        "test/neighboring_engine_test.py",
        "test/quantization_test.py",
        "test/rank_two_test.py",
        "test/reshape_transpose_test.py",
        "test/topk_test.py",
        "test/trt_mode_test.py",
        "test/unary_test.py",
        "test/vgg_block_nchw_test.py",
        "test/vgg_block_test.py",
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
        ":tf_trt_integration_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

cuda_py_test(
    name = "quantization_mnist_test",
    srcs = ["test/quantization_mnist_test.py"],
    data = [
        "test/testdata/mnist/checkpoint",
        "test/testdata/mnist/model.ckpt-46900.data-00000-of-00001",
        "test/testdata/mnist/model.ckpt-46900.index",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # TODO(b/125290478): allow running in at least some OSS configurations.
        "no_pip",
        "no_rocm",
        "no_windows",
        "nomac",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/estimator",
    ],
)
