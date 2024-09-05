# python/ops/memory_tests package

load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_additional_xla_deps_py")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")

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
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
        "@local_xla//xla/service:hlo_proto_py",
    ] + tf_additional_xla_deps_py(),
)

tpu_py_strict_test(
    name = "custom_gradient_memory_test_tpu",
    size = "medium",
    srcs = ["custom_gradient_memory_test.py"],
    main = "custom_gradient_memory_test.py",
    deps = [
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
        "@local_xla//xla/service:hlo_proto_py",
    ] + tf_additional_xla_deps_py(),
)
