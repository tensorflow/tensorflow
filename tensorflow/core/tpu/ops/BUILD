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
        ":sparse_core_ops",
        ":sparse_core_preprocess_ops",
        ":topk_ops",
        ":tpu_compile_op",
        ":tpu_copy_with_dynamic_shape_op",
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
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:status",
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
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_round_robin_op",
    srcs = [
        "tpu_round_robin_op.cc",
    ],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
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
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
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
        "@com_google_absl//absl/status",
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
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_reshard_variables_op",
    srcs = [
        "tpu_reshard_variables_op.cc",
    ],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
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
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:util",
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
        ":tpu_embedding_shape_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "//tensorflow/core/tpu:tpu_embedding_output_layout_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:logging",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sparse_core_preprocess_ops",
    srcs = [
        "sparse_core_preprocess_ops.cc",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sparse_core_ops",
    srcs = [
        "sparse_core_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//xla:util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_copy_with_dynamic_shape_op",
    srcs = [
        "tpu_copy_with_dynamic_shape_op.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/lib/core:status",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:errors",
    ],
    alwayslink = 1,
)
