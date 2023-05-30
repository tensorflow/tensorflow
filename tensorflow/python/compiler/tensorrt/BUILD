# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

# cuda_py_test and cuda_py_tests enable XLA tests by default. We can't
# combine XLA with TensorRT currently and should set
# xla_enable_strict_auto_jit to False to disable XLA tests.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "test/testdata/*",
]))

py_strict_library(
    name = "init_py",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":tf_trt_integration_test_base",  # build_cleaner: keep
        ":trt_convert_py",
    ],
)

py_strict_library(
    name = "trt_convert_py",
    srcs = [
        "trt_convert.py",
        "utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/tf2tensorrt:_pywrap_py_utils",
        "//tensorflow/compiler/tf2tensorrt:trt_ops_loader",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:resource_variable_ops_gen",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/grappler:tf_optimizer",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:lazy_loader",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "tf_trt_integration_test_base",
    srcs = ["//tensorflow/python/compiler/tensorrt/test:tf_trt_integration_test_base_srcs"],
    srcs_version = "PY3",
    deps = [
        ":trt_convert_py",
        "//tensorflow/compiler/tf2tensorrt:_pywrap_py_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
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

cuda_py_strict_test(
    name = "trt_convert_test",
    srcs = ["trt_convert_test.py"],
    data = [
        "//tensorflow/python/compiler/tensorrt/test:trt_convert_test_data",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_pip",
        "nomac",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":trt_convert_py",
        "//tensorflow/compiler/tf2tensorrt:_pywrap_py_utils",
        "//tensorflow/compiler/tf2tensorrt:trt_engine_instance_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/compiler/tensorrt/test:test_utils",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "//tensorflow/python/tools:saved_model_utils",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/util:lazy_loader",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "quantization_mnist_test",
    srcs = ["//tensorflow/python/compiler/tensorrt/test:quantization_mnist_test_srcs"],
    data = [
        "//tensorflow/python/compiler/tensorrt/test:quantization_mnist_test_data",
    ],
    python_version = "PY3",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # TODO(b/125290478): allow running in at least some OSS configurations.
        "no_pip",
        "no_rocm",
        "no_windows",
        "nomac",
        "requires-net:external",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base",
        ":trt_convert_py",
        "//tensorflow/compiler/tf2tensorrt:_pywrap_py_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/estimator:run_config",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/keras:metrics",
        "//tensorflow/python/layers",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/training:adam",
        "//tensorflow/python/training:checkpoint_management",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/training:training_util",
    ],
)
