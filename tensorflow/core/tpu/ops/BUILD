load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        ":tpu_embedding_ops",
        ":tpu_execute_op",
        ":tpu_handle_to_key_op",
        ":tpu_partitioned_ops",
        ":tpu_reshard_variables_op",
        ":tpu_round_robin_op",
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
    name = "tpu_round_robin_op",
    srcs = [
        "tpu_round_robin_op.cc",
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
    name = "tpu_handle_to_key_op",
    srcs = [
        "tpu_handle_to_key_op.cc",
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

cc_library(
    name = "tpu_embedding_shape_util",
    srcs = [
        "tpu_embedding_shape_util.cc",
    ],
    hdrs = [
        "tpu_embedding_shape_util.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_embedding_ops",
    srcs = [
        "tpu_embedding_ops.cc",
    ],
    hdrs = [
        "tpu_embedding_ops.h",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/core/tpu:tpu_embedding_output_layout_utils",
        "//tensorflow/core/tpu/ops:tpu_embedding_shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)
