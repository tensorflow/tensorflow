# Description:
# Protocol buffer generation for tensorflow/core/framework.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

package(default_visibility = ["//tensorflow/go:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# -----------------------------------------------------------------------------
# Framework protos are assigned unique import paths, thus individual libraries
# https://github.com/tensorflow/tensorflow/commit/4221d1a

go_proto_library(
    name = "allocation_description_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/allocation_description_go_proto",
    protos = ["//tensorflow/core/framework:allocation_description_proto"],
)

go_proto_library(
    name = "api_def_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/api_def_go_proto",
    protos = ["//tensorflow/core/framework:api_def_proto"],
    deps = [
        ":attr_value_go_proto",
        ":resource_handle_go_proto",
        ":tensor_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "attr_value_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/attr_value_go_proto",
    protos = ["//tensorflow/core/framework:attr_value_proto"],
    deps = [
        ":resource_handle_go_proto",
        ":tensor_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "cost_graph_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/cost_graph_go_proto",
    protos = ["//tensorflow/core/framework:cost_graph_proto"],
    deps = [
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "dataset_metdata_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/dataset_metadata_go_proto",
    protos = ["//tensorflow/core/framework:dataset_metadata_proto"],
)

go_proto_library(
    name = "dataset_options_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/dataset_options_go_proto",
    protos = ["//tensorflow/core/framework:dataset_options_proto"],
    deps = [
        ":model_go_proto",
    ],
)

go_proto_library(
    name = "device_attributes_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/device_attributes_go_proto",
    protos = ["//tensorflow/core/framework:device_attributes_proto"],
)

go_proto_library(
    name = "full_type_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/full_type_go_proto",
    protos = ["//tensorflow/core/framework:full_type_proto"],
)

go_proto_library(
    name = "function_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/function_go_proto",
    protos = ["//tensorflow/core/framework:function_proto"],
    deps = [
        ":attr_value_go_proto",
        ":node_def_go_proto",
        ":op_def_go_proto",
        ":resource_handle_go_proto",
        ":tensor_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "graph_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/graph_go_proto",
    protos = ["//tensorflow/core/framework:graph_proto"],
    deps = [
        ":attr_value_go_proto",
        ":function_go_proto",
        ":node_def_go_proto",
        ":op_def_go_proto",
        ":resource_handle_go_proto",
        ":tensor_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
        ":versions_go_proto",
    ],
)

go_proto_library(
    name = "graph_transfer_info_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/graph_transfer_info_go_proto",
    protos = ["//tensorflow/core/framework:graph_transfer_info_proto"],
    deps = [
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "kernel_def_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/kernel_def_go_proto",
    protos = ["//tensorflow/core/framework:kernel_def_proto"],
    deps = [
        ":attr_value_go_proto",
        ":resource_handle_go_proto",
        ":tensor_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "log_memory_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/log_memory_go_proto",
    protos = ["//tensorflow/core/framework:log_memory_proto"],
    deps = [
        ":allocation_description_go_proto",
        ":tensor_description_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "model_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/model_go_proto",
    protos = ["//tensorflow/core/framework:model_proto"],
)

go_proto_library(
    name = "node_def_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/node_def_go_proto",
    protos = ["//tensorflow/core/framework:node_def_proto"],
    deps = [
        ":attr_value_go_proto",
        ":full_type_go_proto",
        ":resource_handle_go_proto",
        ":tensor_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "op_def_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/op_def_go_proto",
    protos = ["//tensorflow/core/framework:op_def_proto"],
    deps = [
        ":attr_value_go_proto",
        ":full_type_go_proto",
        ":resource_handle_go_proto",
        ":tensor_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "optimized_function_graph_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/optimized_function_graph_go_proto",
    protos = ["//tensorflow/core/framework:optimized_function_graph_proto"],
    deps = [
        ":graph_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "reader_base_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/reader_base_go_proto",
    protos = ["//tensorflow/core/framework:reader_base_proto"],
)

go_proto_library(
    name = "resource_handle_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/resource_handle_go_proto",
    protos = ["//tensorflow/core/framework:resource_handle_proto"],
    deps = [
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "step_stats_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/step_stats_go_proto",
    protos = ["//tensorflow/core/framework:step_stats_proto"],
    deps = [
        ":allocation_description_go_proto",
        ":tensor_description_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "summary_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/summary_go_proto",
    protos = ["//tensorflow/core/framework:summary_proto"],
    deps = [
        ":tensor_go_proto",
        "//tensorflow/go/tsl/protobuf:for_core_protos_go_proto",
    ],
)

go_proto_library(
    name = "tensor_description_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_description_go_proto",
    protos = ["//tensorflow/core/framework:tensor_description_proto"],
    deps = [
        ":allocation_description_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "tensor_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_go_proto",
    protos = ["//tensorflow/core/framework:tensor_proto"],
    deps = [
        ":resource_handle_go_proto",
        ":tensor_shape_go_proto",
        ":types_go_proto",
    ],
)

go_proto_library(
    name = "tensor_shape_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_shape_go_proto",
    protos = ["//tensorflow/core/framework:tensor_shape_proto"],
)

go_proto_library(
    name = "tensor_slice_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_slice_go_proto",
    protos = ["//tensorflow/core/framework:tensor_slice_proto"],
)

go_proto_library(
    name = "types_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto",
    protos = ["//tensorflow/core/framework:types_proto"],
)

go_proto_library(
    name = "variable_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/variable_go_proto",
    protos = ["//tensorflow/core/framework:variable_proto"],
)

go_proto_library(
    name = "versions_go_proto",
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/versions_go_proto",
    protos = ["//tensorflow/core/framework:versions_proto"],
)
