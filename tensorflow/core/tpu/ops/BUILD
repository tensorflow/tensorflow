load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "ops",
    linkstatic = 1,
    deps = [
        ":host_compute_ops",
        ":topk_ops",
        ":tpu_compile_op",
        ":tpu_execute_op",
        ":tpu_partitioned_ops",
        ":tpu_reshard_variables_op",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_partitioned_ops",
    srcs = [
        "tpu_partitioned_input_op.cc",
        "tpu_partitioned_output_op.cc",
    ],
    linkstatic = 1,
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
    linkstatic = 1,
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
    linkstatic = 1,
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
    linkstatic = 1,
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
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_reshard_variables_op",
    srcs = [
        "tpu_reshard_variables_op.cc",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)
