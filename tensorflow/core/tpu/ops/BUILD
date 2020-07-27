package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tpu_partitioned_ops",
    srcs = [
        "tpu_partitioned_input_op.cc",
        "tpu_partitioned_output_op.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_compile_op",
    srcs = [
        "tpu_compile_op.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_execute_op",
    srcs = [
        "tpu_execute_op.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "host_compute_ops",
    srcs = [
        "host_compute_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "topk_ops",
    srcs = [
        "topk_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)
