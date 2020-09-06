load("//tensorflow:tensorflow.bzl", "tf_cc_test")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "auto_clustering_test_helper",
    testonly = True,
    srcs = ["auto_clustering_test_helper.cc"],
    hdrs = ["auto_clustering_test_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:compilation_passes",
        "//tensorflow/compiler/jit:jit_compilation_passes",
        "//tensorflow/compiler/jit:xla_cluster_util",
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/tools/optimization:optimization_pass_runner_lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "auto_clustering_test",
    srcs = ["auto_clustering_test.cc"],
    data = [
        "keras_imagenet_main.golden_summary",
        "keras_imagenet_main.pbtxt",
        "keras_imagenet_main_graph_mode.golden_summary",
        "keras_imagenet_main_graph_mode.pbtxt",
        "opens2s_gnmt_mixed_precision.golden_summary",
        "opens2s_gnmt_mixed_precision.pbtxt.gz",
    ],
    tags = ["config-cuda-only"],
    deps = [
        ":auto_clustering_test_helper",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)
