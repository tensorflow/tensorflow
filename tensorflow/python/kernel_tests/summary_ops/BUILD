# Tests of TensorFlow kernels written using the Python API.

load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "summary_ops_test",
    size = "medium",
    srcs = ["summary_ops_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/module",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

cuda_py_strict_test(
    name = "summary_v1_audio_op_test",
    size = "small",
    srcs = ["summary_v1_audio_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/summary:summary_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "summary_v1_image_op_test",
    size = "small",
    srcs = ["summary_v1_image_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/summary:summary_py",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "summary_v1_ops_test",
    size = "small",
    srcs = ["summary_v1_ops_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/summary:summary_py",
    ],
)

tf_py_strict_test(
    name = "summary_v1_tensor_op_test",
    size = "small",
    srcs = ["summary_v1_tensor_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/summary:summary_py",
        "//third_party/py/numpy",
    ],
)
