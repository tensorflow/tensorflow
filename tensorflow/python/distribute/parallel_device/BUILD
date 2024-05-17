load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow/core/platform:distribute.bzl", "distribute_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# Pybind rules must live in tensorflow/python due to header rule visibility.
exports_files(
    ["pywrap_parallel_device.cc"],
    visibility = ["//tensorflow/python:__pkg__"],
)

py_strict_library(
    name = "parallel_device",
    srcs = ["parallel_device.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tensorflow_numerics/extensions:__pkg__",
    ],
    deps = [
        "//tensorflow/python:_pywrap_parallel_device",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:variable_utils",
    ],
)

distribute_py_strict_test(
    name = "parallel_device_test",
    srcs = ["parallel_device_test.py"],
    # We don't only use TPU Runtime on parallel device, we use c_api_tfrt instead.
    disable_tpu_use_tfrt = True,
    python_version = "PY3",
    tags = [
        # Dependencies aren't otherwise included in the pip package yet.
        "no_pip",
        "notpu",  # TODO(b/210159281)
    ],
    deps = [
        ":parallel_device",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:collective_ops",
        "//tensorflow/python/ops:control_flow_switch_case",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:stateful_random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)
