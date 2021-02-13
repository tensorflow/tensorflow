# XLA Ops for TPUs

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "xla_ops",
    srcs = [
        "get_item_op.cc",
        "host_compute_ops.cc",
        "index_ops.cc",
        "infeed_op.cc",
        "inplace_ops.cc",
        "outfeed_ops.cc",
        "segment_reduction_ops.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:sharding_util",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:if_op",
        "//tensorflow/compiler/tf2xla/kernels:while_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla/lib:scatter",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:side_effect_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:comparators",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu/kernels:cross_replica_ops",
        "//tensorflow/stream_executor/tpu:c_api_conversions",
        "//tensorflow/stream_executor/tpu:c_api_decl",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
