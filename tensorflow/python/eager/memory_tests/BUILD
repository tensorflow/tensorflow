load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

# NOTE: Do not add sharding to these tests. If tests run concurrently, they
# seem to confuse the memory_profiler, and the tests begin to flake. Add new
# test files as needed.

cuda_py_test(
    name = "memory_test",
    size = "medium",
    srcs = ["memory_test.py"],
    additional_deps = [
        "//tensorflow/python/keras",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "@six_archive//:six",
    ],
    tags = [
        "optonly",  # The test is too slow in non-opt mode
    ],
    xla_enable_strict_auto_jit = True,
)

cuda_py_test(
    name = "remote_memory_test",
    size = "medium",
    srcs = ["remote_memory_test.py"],
    additional_deps = [
        ":memory_test",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "optonly",  # The test is too slow in non-opt mode
    ],
    xla_enable_strict_auto_jit = True,
)
