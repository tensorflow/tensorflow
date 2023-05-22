load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/function/runtime_client:__subpackages__",
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
        "//tensorflow/core/function/capture:restore_captures",
        "//tensorflow/core/function/runtime_client:runtime_client_py",
        "//tensorflow/python:default_gradient",
        "//tensorflow/python:gradients",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function_def_to_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
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
        "//tensorflow/python:while_loop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/module",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "@absl_py//absl/testing:parameterized",
    ],
)
