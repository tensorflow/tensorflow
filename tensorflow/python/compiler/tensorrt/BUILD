# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

load("//tensorflow:tensorflow.default.bzl", "cuda_py_test")

# cuda_py_test enable XLA tests by default. We can't
# combine XLA with TensorRT currently and should set
# xla_enable_strict_auto_jit to False to disable XLA tests.

# Loading TFTRT_PY_TEST_TAGS - Common set of tags used accross TF-TRT modules & tests
load("//tensorflow/python/compiler/tensorrt:tensorrt.bzl", "TFTRT_PY_TEST_TAGS")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

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
        "//tensorflow/python:platform",
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
    name = "tf_trt_integration_test_base",
    srcs = ["//tensorflow/python/compiler/tensorrt/test:tf_trt_integration_test_base_srcs"],
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
        "//tensorflow/python/compiler/tensorrt/test:trt_convert_test_data",
    ],
    python_version = "PY3",
    tags = ["no_pip"] + TFTRT_PY_TEST_TAGS,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":trt_convert_py",
        "//tensorflow/compiler/tf2tensorrt:trt_engine_instance_proto_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:graph_util",
        "//tensorflow/python/compiler/tensorrt/test:test_utils",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "//tensorflow/python/tools:freeze_graph_lib",
        "//tensorflow/python/tools:saved_model_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)
