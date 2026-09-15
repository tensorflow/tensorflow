load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# Kernel implementations for KDNN-backed ops. Each kernel here is a
# drop-in replacement for the corresponding public op (e.g. _KdnnSigmoid
# replaces Sigmoid). The Grappler remapper substitutes them in when
# IsKDNNEnabled() returns true AND the dtype is supported.

# Shared template for unary element-wise activations. Add new
# activation op instantiations by adding a `KdnnUnaryOp<KDNN::XXX, T>`
# registration in the corresponding .cc file.
cc_library(
    name = "kdnn_unary_op",
    hdrs = ["kdnn_unary_op.h"],
    deps = [
        "//third_party/KDNN:kdnn",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:ops_util",
    ],
)

cc_library(
    name = "kdnn_sigmoid_op",
    srcs = ["kdnn_sigmoid_op.cc"],
    hdrs = ["kdnn_unary_op.h"],
    deps = [
        ":kdnn_unary_op",
        "//third_party/KDNN:kdnn",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util:port",
    ],
)

# Microbenchmark comparing _KdnnSigmoid to default CPU Sigmoid. Intended to
# be run on aarch64 + libkdnn.so to collect the speedup figures referenced
# in PR #124543 review feedback. On every other platform the benchmark is
# a no-op that reports "KDNN unavailable on this platform" and exits
# cleanly, so it is safe to keep this target in the default build graph.
#
# We use `cc_test` rather than `cc_library` because GoogleTest's BENCHMARK()
# macro registers a TEST() entry point at static init time; the registered
# test binary is what `bazel test` discovers.
cc_test(
    name = "kdnn_sigmoid_benchmark_test",
    size = "small",
    srcs = ["kdnn_sigmoid_benchmark.cc"],
    deps = [
        ":kdnn_sigmoid_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/util:port",
    ],
)
