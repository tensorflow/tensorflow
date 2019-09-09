load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

# NOTE: Do not add sharding to these tests. If tests run concurrently, they
# seem to confuse the memory_profiler, and the tests begin to flake. Add new
# test files as needed.

py_library(
    name = "memory_test_util",
    srcs = ["memory_test_util.py"],
    visibility = ["//tensorflow:internal"],
    deps = ["@six_archive//:six"],
)

cuda_py_test(
    name = "memory_test",
    size = "medium",
    srcs = ["memory_test.py"],
    additional_deps = [
        ":memory_test_util",
        "//tensorflow/python/keras",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
    tags = [
        "manual",
        "no_oss",
        "notap",  #TODO(b/140640597): this test is flaky at the moment
        "optonly",  # The test is too slow in non-opt mode
    ],
    # TODO(b/140065350): Re-enable
    xla_enable_strict_auto_jit = False,
)

cuda_py_test(
    name = "remote_memory_test",
    size = "medium",
    srcs = ["remote_memory_test.py"],
    additional_deps = [
        ":memory_test_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "optonly",  # The test is too slow in non-opt mode
    ],
    xla_enable_strict_auto_jit = False,  # b/140261762
)
