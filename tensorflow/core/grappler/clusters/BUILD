licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

config_setting(
    name = "xsmm",
    licenses = ["notice"],
    values = {
        "define": "tensorflow_xsmm=1",
    },
)

tf_cuda_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_id",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ] + select({
        ":xsmm": ["@libxsmm_archive//:xsmm_avx"],
        "//conditions:default": [],
    }),
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    linkstatic = if_cuda(1, 0),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":utils",
        "//tensorflow/core:gpu_id",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "cluster",
    srcs = ["cluster.cc"],
    hdrs = [
        "cluster.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "virtual_cluster",
    srcs = ["virtual_cluster.cc"],
    hdrs = [
        "virtual_cluster.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/costs:op_level_cost_estimator",
        "//tensorflow/core/grappler/costs:virtual_scheduler",
    ],
)

tf_cc_test(
    name = "virtual_cluster_test",
    srcs = ["virtual_cluster_test.cc"],
    deps = [
        ":virtual_cluster",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "single_machine",
    srcs = ["single_machine.cc"],
    hdrs = [
        "single_machine.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":utils",
        "//tensorflow/cc:coordinator",
        "//tensorflow/cc:queue_runner",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_id",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cc_test(
    name = "single_machine_test",
    srcs = ["single_machine_test.cc"],
    args = ["--heap_check=local"],  # The GPU tracer leaks memory
    tags = [
        "no_cuda_on_cpu_tap",
        "no_gpu",
    ],
    deps = [
        ":single_machine",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)
