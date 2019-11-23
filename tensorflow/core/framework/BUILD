load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_proto_library",
    "tf_pyclif_proto_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_generate_proto_text_sources",
)

package(
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Export all files for which we do not yet provide a dedicated build rule.
# This avoids breading all the rules in tensorflow/core/BUILD.
exports_files(
    srcs = [
        "allocator.cc",
        "allocator.h",
        "allocator_registry.cc",
        "allocator_registry.h",
        "attr_value_util.cc",
        "attr_value_util.h",
        "bfloat16.cc",
        "bfloat16.h",
        "bounds_check.h",
        "cancellation.cc",
        "cancellation.h",
        "collective.cc",
        "collective.h",
        "common_shape_fns.cc",
        "common_shape_fns.h",
        "control_flow.h",
        "cpu_allocator_impl.cc",
        "dataset.cc",
        "dataset.h",
        "dataset_stateful_op_whitelist.h",
        "device_base.cc",
        "device_base.h",
        "fake_input.cc",
        "fake_input.h",
        "function.cc",
        "function.h",
        "function_handle_cache.cc",
        "function_handle_cache.h",
        "function_testlib.cc",
        "function_testlib.h",
        "graph_def_util.cc",
        "graph_def_util.h",
        "graph_to_functiondef.cc",
        "graph_to_functiondef.h",
        "kernel_def_builder.cc",
        "kernel_def_builder.h",
        "kernel_def_util.cc",
        "kernel_def_util.h",
        "load_library.cc",
        "log_memory.cc",
        "log_memory.h",
        "logging.cc",
        "logging.h",
        "lookup_interface.cc",
        "lookup_interface.h",
        "memory_types.cc",
        "memory_types.h",
        "model.cc",
        "model.h",
        "node_def_builder.cc",
        "node_def_builder.h",
        "node_def_util.cc",
        "node_def_util.h",
        "numeric_op.h",
        "numeric_types.h",
        "op.cc",
        "op.h",
        "op_def_builder.cc",
        "op_def_builder.h",
        "op_def_util.cc",
        "op_def_util.h",
        "op_gen_lib.cc",
        "op_gen_lib.h",
        "op_kernel.cc",
        "op_kernel.h",
        "op_segment.cc",
        "op_segment.h",
        "ops_util.cc",
        "ops_util.h",
        "partial_tensor_shape.h",
        "queue_interface.h",
        "reader_base.cc",
        "reader_base.h",
        "reader_interface.h",
        "reader_op_kernel.h",
        "register_types.h",
        "register_types_traits.h",
        "rendezvous.cc",
        "rendezvous.h",
        "resource_handle.cc",
        "resource_handle.h",
        "resource_mgr.cc",
        "resource_mgr.h",
        "resource_op_kernel.h",
        "resource_var.h",
        "run_handler.cc",
        "run_handler.h",
        "run_handler_util.cc",
        "run_handler_util.h",
        "selective_registration.h",
        "session_state.h",
        "shape_inference.cc",
        "shape_inference.h",
        "shape_inference_testutil.cc",
        "shape_inference_testutil.h",
        "shared_ptr_variant.h",
        "stats_aggregator.h",
        "tensor.cc",
        "tensor.h",
        "tensor_reference.cc",
        "tensor_reference.h",
        "tensor_shape.cc",
        "tensor_shape.h",
        "tensor_slice.cc",
        "tensor_slice.h",
        "tensor_testutil.cc",
        "tensor_testutil.h",
        "tensor_types.h",
        "tensor_util.cc",
        "tensor_util.h",
        "thread_factory.h",
        "tracking_allocator.cc",
        "tracking_allocator.h",
        "type_index.h",
        "type_traits.h",
        "typed_allocator.cc",
        "typed_allocator.h",
        "types.cc",
        "types.h",
        "unique_tensor_references.cc",
        "unique_tensor_references.h",
        "variant.cc",
        "variant.h",
        "variant_encode_decode.h",
        "variant_op_registry.cc",
        "variant_op_registry.h",
        "variant_tensor_data.cc",
        "variant_tensor_data.h",
        "versions.cc",
        "versions.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

# List of exported test source files that do not yet have local build rules.
exports_files(
    srcs = [
        "allocator_test.cc",
        "attr_value_util_test.cc",
        "bfloat16_test.cc",
        "cancellation_test.cc",
        "common_shape_fns_test.cc",
        "dataset_test.cc",
        "device_base_test.cc",
        "function_test.cc",
        "graph_def_util_test.cc",
        "graph_to_functiondef_test.cc",
        "kernel_def_builder_test.cc",
        "kernel_def_util_test.cc",
        "memory_types_test.cc",
        "model_test.cc",
        "node_def_builder_test.cc",
        "node_def_util_test.cc",
        "op_compatibility_test.cc",
        "op_def_builder_test.cc",
        "op_def_util_test.cc",
        "op_gen_lib_test.cc",
        "op_kernel_test.cc",
        "op_registration_test.cc",
        "op_segment_test.cc",
        "partial_tensor_shape_test.cc",
        "rendezvous_test.cc",
        "resource_mgr_test.cc",
        "resource_op_kernel_test.cc",
        "run_handler_test.cc",
        "run_handler_util_test.cc",
        "shape_inference_test.cc",
        "shape_inference_testutil_test.cc",
        "tensor_shape_test.cc",
        "tensor_slice_test.cc",
        "tensor_test.cc",
        "tensor_testutil_test.cc",
        "tensor_util_test.cc",
        "tracking_allocator_test.cc",
        "types_test.cc",
        "unique_tensor_references_test.cc",
        "variant_op_copy_test.cc",
        "variant_op_registry_test.cc",
        "variant_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

# List of exported proto source files.
exports_files(
    srcs = [
        "allocation_description.proto",
        "api_def.proto",
        "attr_value.proto",
        "cost_graph.proto",
        "device_attributes.proto",
        "function.proto",
        "graph.proto",
        "graph_transfer_info.proto",
        "kernel_def.proto",
        "log_memory.proto",
        "node_def.proto",
        "op_def.proto",
        "reader_base.proto",
        "remote_fused_graph_execute_info.proto",
        "resource_handle.proto",
        "step_stats.proto",
        "summary.proto",
        "tensor.proto",
        "tensor_description.proto",
        "tensor_shape.proto",
        "tensor_slice.proto",
        "types.proto",
        "variable.proto",
        "versions.proto",
    ],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/core:__pkg__",
    ],
)

# The following filegroups are needed since globbing across packages boundaries
# will just fail silently (see 3rd caveat at
# https://docs.bazel.build/versions/master/be/functions.html#glob).

# Files needed for core:framework_internal_impl.
filegroup(
    name = "framework_internal_private_hdrs",
    srcs = glob(
        [
            "**/*.h",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
            "fake_input.*",
            "op_gen_lib.*",
            "reader_base.*",
        ],
    ),
)

filegroup(
    name = "framework_internal_impl_srcs",
    srcs = glob(
        [
            "**/*.cc",
        ],
        exclude = [
            "**/*test*",
            "**/*main.cc",
            "allocator.cc",
            "cpu_allocator_impl.cc",
            "allocator_registry.cc",
            "tracking_allocator.cc",
            "fake_input.*",
            "op_gen_lib.*",
            "reader_base.*",
        ],
    ),
)

# Files needed for core:mobile_srcs_(no|only)_runtime.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = glob(
        [
            "**/*.h",
            "**/*.cc",
        ],
        exclude = [
            "**/*test.*",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
            "debug/**/*",
            "op_gen_*",
            "dataset.*",
            "node_def_util.cc",
            "op_kernel.cc",
        ],
    ),
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = glob(
        [
            "dataset.*",
        ],
        exclude = [
            "**/*test.*",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
        ],
    ) + [
        "node_def_util.cc",
        "op_kernel.cc",
    ],
)

# All framewrok protos are self-contained, i.e. they only import other
# protos from the same package, so we can build the protos here and then
# link them from core:protos_all without circular dependencies.

# Generate the C++ sources for tsome of the protos.
tf_generate_proto_text_sources(
    name = "attr_value_proto_text",
    srcs = [
        "attr_value.proto",
        "resource_handle.proto",
        "tensor.proto",
        "tensor_shape.proto",
        "types.proto",
    ],
    srcs_relative_dir = "tensorflow/core/framework/",
    deps = [
        ":attr_value_proto_cc",
        ":resource_handle_proto_cc",
        ":tensor_proto_cc",
        ":tensor_shape_proto_cc",
        ":types_proto_cc",
        "//tensorflow/core/lib/strings:proto_text_util",
        "//tensorflow/core/lib/strings:scanner",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:types",
    ],
)

tf_pyclif_proto_library(
    name = "cost_graph_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "cost_graph.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "tensor_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "tensor.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "kernel_def_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "kernel_def.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "node_def_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "node_def.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "function_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "function.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "graph_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "graph.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "step_stats_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "step_stats.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "types_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "types.proto",
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "variable_pyclif",
    proto_lib = "//tensorflow/core:protos_all",
    proto_srcfile = "variable.proto",
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "log_memory_proto",
    srcs = ["log_memory.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":allocation_description_proto",
        ":tensor_description_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "versions_proto",
    srcs = ["versions.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "graph_proto",
    srcs = ["graph.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":attr_value_proto",
        ":function_proto",
        ":node_def_proto",
        ":op_def_proto",
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
        ":versions_proto",
    ],
)

tf_proto_library(
    name = "node_def_proto",
    srcs = ["node_def.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":attr_value_proto",
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "allocation_description_proto",
    srcs = ["allocation_description.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "tensor_slice_proto",
    srcs = ["tensor_slice.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "tensor_description_proto",
    srcs = ["tensor_description.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":allocation_description_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "device_attributes_proto",
    srcs = ["device_attributes.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "resource_handle_proto",
    srcs = ["resource_handle.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "step_stats_proto",
    srcs = ["step_stats.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":allocation_description_proto",
        ":tensor_description_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "reader_base_proto",
    srcs = ["reader_base.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "kernel_def_proto",
    srcs = ["kernel_def.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":attr_value_proto",
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "op_def_proto",
    srcs = ["op_def.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":attr_value_proto",
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "attr_value_proto",
    srcs = ["attr_value.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "remote_fused_graph_execute_info_proto",
    srcs = ["remote_fused_graph_execute_info.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":attr_value_proto",
        ":function_proto",
        ":graph_proto",
        ":node_def_proto",
        ":op_def_proto",
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
        ":versions_proto",
    ],
)

tf_proto_library(
    name = "tensor_proto",
    srcs = ["tensor.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":resource_handle_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "api_def_proto",
    srcs = ["api_def.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":attr_value_proto",
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "variable_proto",
    srcs = ["variable.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "graph_transfer_info_proto",
    srcs = ["graph_transfer_info.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":types_proto",
    ],
)

tf_proto_library(
    name = "types_proto",
    srcs = ["types.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "cost_graph_proto",
    srcs = ["cost_graph.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "tensor_shape_proto",
    srcs = ["tensor_shape.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "function_proto",
    srcs = ["function.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":attr_value_proto",
        ":node_def_proto",
        ":op_def_proto",
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "summary_proto",
    srcs = ["summary.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "protos_all",
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":allocation_description_proto",
        ":api_def_proto",
        ":attr_value_proto",
        ":cost_graph_proto",
        ":device_attributes_proto",
        ":function_proto",
        ":graph_proto",
        ":graph_transfer_info_proto",
        ":kernel_def_proto",
        ":log_memory_proto",
        ":node_def_proto",
        ":op_def_proto",
        ":reader_base_proto",
        ":remote_fused_graph_execute_info_proto",
        ":resource_handle_proto",
        ":step_stats_proto",
        ":summary_proto",
        ":tensor_description_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":tensor_slice_proto",
        ":types_proto",
        ":variable_proto",
        ":versions_proto",
    ],
)
