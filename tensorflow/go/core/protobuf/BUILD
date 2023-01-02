# Description:
# Protocol buffer generation for tensorflow/core/protobuf.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

package(default_visibility = ["//tensorflow/go:__subpackages__"])

licenses(["notice"])  # Apache 2.0

go_proto_library(
    name = "for_core_protos_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto",
    protos = [
        "//tensorflow/core/protobuf:error_codes_proto_impl",
        "//tensorflow/core/protobuf:for_core_protos",
    ],
    deps = [
        "//tensorflow/go/core/framework:allocation_description_go_proto",
        "//tensorflow/go/core/framework:api_def_go_proto",
        "//tensorflow/go/core/framework:attr_value_go_proto",
        "//tensorflow/go/core/framework:cost_graph_go_proto",
        "//tensorflow/go/core/framework:dataset_options_go_proto",
        "//tensorflow/go/core/framework:device_attributes_go_proto",
        "//tensorflow/go/core/framework:full_type_go_proto",
        "//tensorflow/go/core/framework:function_go_proto",
        "//tensorflow/go/core/framework:graph_go_proto",
        "//tensorflow/go/core/framework:graph_transfer_info_go_proto",
        "//tensorflow/go/core/framework:kernel_def_go_proto",
        "//tensorflow/go/core/framework:log_memory_go_proto",
        "//tensorflow/go/core/framework:model_go_proto",
        "//tensorflow/go/core/framework:node_def_go_proto",
        "//tensorflow/go/core/framework:op_def_go_proto",
        "//tensorflow/go/core/framework:reader_base_go_proto",
        "//tensorflow/go/core/framework:resource_handle_go_proto",
        "//tensorflow/go/core/framework:step_stats_go_proto",
        "//tensorflow/go/core/framework:summary_go_proto",
        "//tensorflow/go/core/framework:tensor_description_go_proto",
        "//tensorflow/go/core/framework:tensor_go_proto",
        "//tensorflow/go/core/framework:tensor_shape_go_proto",
        "//tensorflow/go/core/framework:tensor_slice_go_proto",
        "//tensorflow/go/core/framework:types_go_proto",
        "//tensorflow/go/core/framework:variable_go_proto",
        "//tensorflow/go/core/framework:versions_go_proto",
        "//tensorflow/go/tsl/protobuf:for_core_protos_go_proto",
        "@com_github_golang_protobuf//ptypes/any",
        "@com_github_golang_protobuf//ptypes/wrappers",
    ],
)
