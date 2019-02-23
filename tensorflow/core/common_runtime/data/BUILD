package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_protos_all")

cc_library(
    name = "standalone",
    srcs = ["standalone.cc"],
    hdrs = ["standalone.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
    ],
)

tf_cc_test(
    name = "standalone_test",
    srcs = ["standalone_test.cc"],
    deps = [
        ":standalone",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + tf_protos_all(),
)
