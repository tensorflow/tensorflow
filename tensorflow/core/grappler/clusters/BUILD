licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cuda_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
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
    hdrs = [
        "utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ] + select({
        ":xsmm": ["@libxsmm_archive//:xsmm_avx"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "cluster",
    srcs = ["cluster.cc"],
    hdrs = [
        "cluster.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
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
        "//tensorflow/core:protos_all_cc",
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
        "//tensorflow/core:direct_session",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/kernels:ops_util",
    ],
)

cc_test(
    name = "single_machine_test",
    srcs = ["single_machine_test.cc"],
    args = ["--heap_check=local"],  # The GPU tracer leaks memory
    deps = [
        ":single_machine",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)
