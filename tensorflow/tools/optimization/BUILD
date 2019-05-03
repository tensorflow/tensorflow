# Description:
#   Utilities that perform useful transformations on graphs

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cuda_library",
)

exports_files(["LICENSE"])

tf_cuda_library(
    name = "optimization_pass_runner_lib",
    srcs = ["optimization_pass_runner.cc"],
    hdrs = ["optimization_pass_runner.h"],
    deps = [
        "//tensorflow/contrib:contrib_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_binary(
    name = "gpu_optimization_pass_runner",
    srcs = ["gpu_optimization_pass_runner_main.cc"],
    deps = [
        ":optimization_pass_runner_lib",
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/contrib:contrib_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "@com_google_absl//absl/strings",
    ],
)
