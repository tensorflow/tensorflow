# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    glob(["*test.py"]),
)

py_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
)

py_strict_library(
    name = "tf_trt_integration_test_base_srcs",
    srcs = ["tf_trt_integration_test_base.py"],
    visibility = ["//tensorflow/python/compiler/tensorrt:__pkg__"],
    deps = [
        "//tensorflow/compiler/tf2tensorrt:_pywrap_py_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "//tensorflow/python/tools:saved_model_utils",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
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

base_tags = [
    "no_cuda_on_cpu_tap",
    "no_rocm",
    "no_windows",
    "nomac",
]

cuda_py_strict_test(
    name = "base_test",
    srcs = ["base_test.py"],
    python_version = "PY3",
    tags = base_tags + ["no_oss"],  # TODO(b/165611343): Need to address the failures for CUDA 11 in OSS build.
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "conv2d_test",
    srcs = ["conv2d_test.py"],
    python_version = "PY3",
    tags = base_tags + ["no_oss"],  # b/198501457
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "combined_nms_test",
    srcs = ["combined_nms_test.py"],
    python_version = "PY3",
    tags = base_tags + ["no_oss"],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "tf_function_test",
    srcs = ["tf_function_test.py"],
    python_version = "PY3",
    tags = base_tags + [
        "manual",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "no_oss",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "notap",  # TODO(b/231239602): re-enable once naming issue is resolved.
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/util:compat",
    ],
)

cuda_py_strict_test(
    name = "batch_matmul_test",
    srcs = ["batch_matmul_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "biasadd_matmul_test",
    srcs = ["biasadd_matmul_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "binary_tensor_weight_broadcast_test",
    srcs = ["binary_tensor_weight_broadcast_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

cuda_py_strict_test(
    name = "bool_test",
    srcs = ["bool_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "cast_test",
    srcs = ["cast_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "concatenation_test",
    srcs = ["concatenation_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "const_broadcast_test",
    srcs = ["const_broadcast_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "data_dependent_shape_test",
    srcs = ["data_dependent_shape_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "dynamic_input_shapes_test",
    srcs = ["dynamic_input_shapes_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "identity_output_test",
    srcs = ["identity_output_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "int32_test",
    srcs = ["int32_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "lru_cache_test",
    srcs = ["lru_cache_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "memory_alignment_test",
    srcs = ["memory_alignment_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "multi_connection_neighbor_engine_test",
    srcs = ["multi_connection_neighbor_engine_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "neighboring_engine_test",
    srcs = ["neighboring_engine_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "quantization_test",
    srcs = ["quantization_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "rank_two_test",
    srcs = ["rank_two_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "reshape_transpose_test",
    srcs = ["reshape_transpose_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "topk_test",
    srcs = ["topk_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "trt_engine_op_shape_test",
    srcs = ["trt_engine_op_shape_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

cuda_py_strict_test(
    name = "trt_mode_test",
    srcs = ["trt_mode_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "unary_test",
    srcs = ["unary_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "vgg_block_nchw_test",
    srcs = ["vgg_block_nchw_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "vgg_block_test",
    srcs = ["vgg_block_test.py"],
    python_version = "PY3",
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

test_suite(
    name = "tf_trt_integration_test",
    tests = [
        "batch_matmul_test",
        "biasadd_matmul_test",
        "binary_tensor_weight_broadcast_test",
        "bool_test",
        "cast_test",
        "combined_nms_test",
        "concatenation_test",
        "const_broadcast_test",
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
        "tf_function_test",
        "topk_test",
        "trt_engine_op_shape_test",
        "trt_mode_test",
        "unary_test",
        "vgg_block_nchw_test",
        "vgg_block_test",
    ],
)

test_suite(
    name = "tf_trt_integration_test_no_oss",
    tests = [
        "base_test",
        "conv2d_test",
    ],
)
