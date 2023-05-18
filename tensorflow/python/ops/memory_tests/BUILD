# python/ops/memory_tests package

load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_additional_xla_deps_py")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "custom_gradient_memory_test",
    size = "medium",
    srcs = ["custom_gradient_memory_test.py"],
    xla_enable_strict_auto_jit = False,  # XLA are enabled explicitly in XLA memory tests.
    deps = [
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
        "//tensorflow/python/tpu:tpu_strategy_util",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

tpu_py_strict_test(
    name = "custom_gradient_memory_test_tpu",
    size = "medium",
    srcs = ["custom_gradient_memory_test.py"],
    # TODO(b/238830423): This test uses experimental_get_compiler_ir, which is
    # not supported with TFRT (Failed getting HLO text: 'GetCompilerIr is not
    # supported on this device.').
    disable_tfrt = True,
    main = "custom_gradient_memory_test.py",
    deps = [
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
        "//tensorflow/python/tpu:tpu_strategy_util",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)
