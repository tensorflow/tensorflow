load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# NOTE: Do not add sharding to these tests. If tests run concurrently, they
# seem to confuse the memory_profiler, and the tests begin to flake. Add new
# test files as needed.

py_strict_library(
    name = "memory_test_util",
    srcs = ["memory_test_util.py"],
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/python/eager:context"],
)

cuda_py_strict_test(
    name = "memory_test",
    size = "medium",
    srcs = ["memory_test.py"],
    tags = [
        "manual",
        "no_oss",
        "notap",  #TODO(b/140640597): this test is flaky at the moment
        "optonly",  # The test is too slow in non-opt mode
    ],
    # TODO(b/140065350): Re-enable
    xla_enable_strict_auto_jit = False,
    deps = [
        ":memory_test_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
    ],
)

cuda_py_strict_test(
    name = "remote_memory_test",
    size = "medium",
    srcs = ["remote_memory_test.py"],
    tags = [
        "no_gpu",  # TODO(b/168058741): Enable the test for GPU
        "optonly",  # The test is too slow in non-opt mode
    ],
    xla_enable_strict_auto_jit = False,  # b/140261762
    deps = [
        ":memory_test_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/training:server_lib",
    ],
)
