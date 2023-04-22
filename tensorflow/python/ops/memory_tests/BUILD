# python/ops/memory_tests package

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow/core/platform:build_config_root.bzl", "tf_additional_xla_deps_py")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cuda_py_test(
    name = "custom_gradient_memory_test",
    size = "medium",
    srcs = ["custom_gradient_memory_test.py"],
    xla_enable_strict_auto_jit = False,  # XLA are enabled explicitly in XLA memory tests.
    deps = [
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/compiler/xla/service:hlo_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
    ] + tf_additional_xla_deps_py(),
)

tpu_py_test(
    name = "custom_gradient_memory_test_tpu",
    size = "medium",
    srcs = ["custom_gradient_memory_test.py"],
    main = "custom_gradient_memory_test.py",
    deps = [
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/compiler/xla/service:hlo_proto_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/tpu:tpu_strategy_util",
    ] + tf_additional_xla_deps_py(),
)
