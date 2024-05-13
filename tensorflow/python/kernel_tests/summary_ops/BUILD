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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
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
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/platform:client_testlib",
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
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:image_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/platform:client_testlib",
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/summary:summary_py",
    ],
)

tf_py_strict_test(
    name = "summary_v1_tensor_op_test",
    size = "small",
    srcs = ["summary_v1_tensor_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/summary:summary_py",
        "//third_party/py/numpy",
    ],
)
