load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_strict_test")

package(
    default_visibility = [
        "//tensorflow/core/function:__subpackages__",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "transform",
    srcs = [
        "transform.py",
    ],
    srcs_version = "PY3",
    visibility = [
        "//learning/brain/experimental/tfq:__subpackages__",
        "//smartass/brain/configure/compile_service/python:__pkg__",
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/framework:function_proto_py",
        "//tensorflow/core/function:runtime_client_py",
        "//tensorflow/python:default_gradient",
        "//tensorflow/python:gradients",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:function_saved_model_utils",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function_def_to_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/util",
    ],
)

py_strict_test(
    name = "transform_test",
    srcs = ["transform_test.py"],
    python_version = "PY3",
    tags = ["no_oss"],  # TODO(b/219089812)
    deps = [
        ":transform",
        "//tensorflow/core/function/testing:test_pass_py",
        "//tensorflow/python:gradients",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
