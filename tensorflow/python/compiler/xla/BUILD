load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "compiler_py",
    srcs = [
        "__init__.py",
        "jit.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":xla",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "jit_test",
    size = "small",
    srcs = ["jit_test.py"],
    tags = [
        "no_windows",  # TODO(b/171385770)
    ],
    xla_enabled = True,
    deps = [
        ":compiler_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:op_def_registry",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "xla",
    srcs = ["xla.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/jit:xla_ops_py",
        "//tensorflow/compiler/jit/ops:xla_ops_grad",
        "//tensorflow/core:protos_all_py",
        # Do not remove: required to run xla ops on Cloud.
        "//tensorflow/compiler/tf2xla/python:xla",  # build_cleaner: keep
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/distribute:summary_op_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "xla_test",
    srcs = ["xla_test.py"],
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
    deps = [
        ":xla",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/summary:__init__",
        "//tensorflow/python/tpu:tpu_feed",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "jit_compile_test",
    srcs = ["jit_compile_test.py"],
    python_version = "PY3",
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "pjrt_compile_test",
    srcs = ["pjrt_compile_test.py"],
    env = {
        "TF_XLA_FLAGS": "--tf_xla_use_device_api --tf_xla_enable_xla_devices",
    },
    python_version = "PY3",
    tags = [
        "config-cuda-only",
        "gpu",
        "no_oss",
        "requires-gpu-nvidia",
        "xla",
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
    ],
)
