# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# cuda_py_test enable XLA tests by default. We can't
# combine XLA with TensorRT currently and should set
# xla_enable_strict_auto_jit to False to disable XLA tests.

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

cuda_py_test(
    name = "batch_matmul_test",
    srcs = [
        "batch_matmul_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "biasadd_matmul_test",
    srcs = [
        "biasadd_matmul_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "binary_tensor_weight_broadcast_test",
    srcs = [
        "binary_tensor_weight_broadcast_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "cast_test",
    srcs = [
        "cast_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "combined_nms_test",
    srcs = [
        "combined_nms_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",
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

cuda_py_test(
    name = "concatenation_test",
    srcs = [
        "concatenation_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "const_broadcast_test",
    srcs = [
        "const_broadcast_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "dynamic_input_shapes_test",
    srcs = [
        "dynamic_input_shapes_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "identity_output_test",
    srcs = [
        "identity_output_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "int32_test",
    srcs = [
        "int32_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "lru_cache_test",
    srcs = [
        "lru_cache_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "memory_alignment_test",
    srcs = [
        "memory_alignment_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "multi_connection_neighbor_engine_test",
    srcs = [
        "multi_connection_neighbor_engine_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "neighboring_engine_test",
    srcs = [
        "neighboring_engine_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "quantization_test",
    srcs = [
        "quantization_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "rank_two_test",
    srcs = [
        "rank_two_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "reshape_transpose_test",
    srcs = [
        "reshape_transpose_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "tf_function_test",
    srcs = [
        "tf_function_test.py",
    ],
    python_version = "PY3",
    tags = [
        "manual",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "no_cuda_on_cpu_tap",
        "no_oss",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "no_rocm",
        "no_windows",
        "nomac",
        "notap",  # TODO(b/231239602): re-enable once naming issue is resolved.
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/compiler/tensorrt:tf_trt_integration_test_base",
    ],
)

cuda_py_test(
    name = "topk_test",
    srcs = [
        "topk_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "trt_engine_op_shape_test",
    srcs = [
        "trt_engine_op_shape_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "trt_mode_test",
    srcs = [
        "trt_mode_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "unary_test",
    srcs = [
        "unary_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "vgg_block_nchw_test",
    srcs = [
        "vgg_block_nchw_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

cuda_py_test(
    name = "vgg_block_test",
    srcs = [
        "vgg_block_test.py",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
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

test_suite(
    name = "tf_trt_integration_test",
    tests = [
        "batch_matmul_test",
        "biasadd_matmul_test",
        "binary_tensor_weight_broadcast_test",
        "cast_test",
        "combined_nms_test",
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
        "tf_function_test",
        "topk_test",
        "trt_engine_op_shape_test",
        "trt_mode_test",
        "unary_test",
        "vgg_block_nchw_test",
        "vgg_block_test",
    ],
)

cuda_py_test(
    name = "base_test",
    srcs = [
        "base_test.py",  # TODO(b/165611343): Need to address the failures for CUDA 11 in OSS build.
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",
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

cuda_py_test(
    name = "conv2d_test",
    srcs = [
        "conv2d_test.py",  # b/198501457
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",
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

test_suite(
    name = "tf_trt_integration_test_no_oss",
    tests = [
        "base_test",
        "conv2d_test",
    ],
)
