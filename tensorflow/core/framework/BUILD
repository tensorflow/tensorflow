load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_kernel_tests_linkstatic",
    "tf_proto_library",
    "tf_pyclif_proto_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
    "tf_generate_proto_text_sources",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Export all header files for which we do not yet provide a dedicated build
# rule. This avoids breading all the rules in tensorflow/core/BUILD.
exports_files(
    srcs = [
        "allocator_registry.h",
        "cancellation.h",
        "collective.h",
        "control_flow.h",
        "dataset.h",
        "dataset_stateful_op_whitelist.h",
        "device_base.h",
        "function.h",
        "function_handle_cache.h",
        "graph_def_util.h",
        "graph_to_functiondef.h",
        "kernel_def_builder.h",
        "kernel_def_util.h",
        "logging.h",
        "lookup_interface.h",
        "memory_types.h",
        "metrics.h",
        "model.h",
        "node_def_builder.h",
        "numeric_op.h",
        "numeric_op_base.h",
        "op_kernel.h",
        "op_requires.h",
        "op_segment.h",
        "ops_util.h",
        "partial_tensor_shape.h",
        "queue_interface.h",
        "reader_interface.h",
        "reader_op_kernel.h",
        "register_types_traits.h",
        "rendezvous.h",
        "resource_mgr.h",
        "resource_op_kernel.h",
        "resource_var.h",
        "run_handler.h",
        "run_handler_util.h",
        "session_state.h",
        "shared_ptr_variant.h",
        "stats_aggregator.h",
        "tensor_reference.h",
        "tensor_slice.h",
        "tensor_util.h",
        "thread_factory.h",
        "tracking_allocator.h",
        "versions.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/common_runtime:__pkg__",
    ],
)

# List of exported test source files that do not yet have local build rules.
exports_files(
    srcs = [
        "op_gen_lib_test.cc",
        "op_segment_test.cc",
        "run_handler_test.cc",
        "run_handler_util_test.cc",
        "variant_op_copy_test.cc",
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

exports_files(
    [
        "attr_value_util.h",
        "common_shape_fns.h",
        "kernel_shape_util.h",
        "node_def_util.h",
        "node_properties.h",
        "op.h",
        "op_def_builder.h",
        "op_def_util.h",
        "selective_registration.h",
        "shape_inference.h",
    ],
    visibility = ["//tensorflow/core:__subpackages__"],
)

# The following filegroups are needed since globbing across packages boundaries
# will just fail silently (see 3rd caveat at
# https://docs.bazel.build/versions/master/be/functions.html#glob).

# Files needed for core:framework_internal_impl.
filegroup(
    name = "framework_internal_private_hdrs",
    srcs = [
        "allocator.h",
        "allocator_registry.h",
        "attr_value_util.h",
        "bfloat16.h",
        "bounds_check.h",
        "cancellation.h",
        "collective.h",
        "common_shape_fns.h",
        "control_flow.h",
        "dataset.h",
        "dataset_stateful_op_whitelist.h",
        "device_base.h",
        "function.h",
        "function_handle_cache.h",
        "graph_def_util.h",
        "graph_to_functiondef.h",
        "kernel_def_builder.h",
        "kernel_def_util.h",
        "kernel_shape_util.h",
        "local_rendezvous.h",
        "log_memory.h",
        "logging.h",
        "lookup_interface.h",
        "memory_types.h",
        "metrics.h",
        "model.h",
        "node_def_builder.h",
        "node_def_util.h",
        "node_properties.h",
        "numeric_op.h",
        "numeric_op_base.h",
        "numeric_types.h",
        "op.h",
        "op_def_builder.h",
        "op_def_util.h",
        "op_kernel.h",
        "op_requires.h",
        "op_segment.h",
        "ops_util.h",
        "partial_tensor_shape.h",
        "queue_interface.h",
        "reader_interface.h",
        "reader_op_kernel.h",
        "register_types.h",
        "register_types_traits.h",
        "rendezvous.h",
        "resource_handle.h",
        "resource_mgr.h",
        "resource_op_kernel.h",
        "resource_var.h",
        "run_handler.h",
        "run_handler_util.h",
        "selective_registration.h",
        "session_state.h",
        "shape_inference.h",
        "shared_ptr_variant.h",
        "stats_aggregator.h",
        "tensor.h",
        "tensor_reference.h",
        "tensor_shape.h",
        "tensor_slice.h",
        "tensor_types.h",
        "tensor_util.h",
        "thread_factory.h",
        "tracking_allocator.h",
        "type_index.h",
        "type_traits.h",
        "typed_allocator.h",
        "types.h",
        "variant.h",
        "variant_encode_decode.h",
        "variant_op_registry.h",
        "variant_tensor_data.h",
        "versions.h",
    ],
)

filegroup(
    name = "framework_internal_impl_srcs",
    srcs = [
        "cancellation.cc",
        "collective.cc",
        "dataset.cc",
        "device_base.cc",
        "function.cc",
        "function_handle_cache.cc",
        "graph_def_util.cc",
        "graph_to_functiondef.cc",
        "kernel_def_builder.cc",
        "kernel_def_util.cc",
        "load_library.cc",
        "local_rendezvous.cc",
        "logging.cc",
        "lookup_interface.cc",
        "memory_types.cc",
        "metrics.cc",
        "model.cc",
        "node_def_builder.cc",
        "op_kernel.cc",
        "op_segment.cc",
        "ops_util.cc",
        "rendezvous.cc",
        "resource_mgr.cc",
        "run_handler.cc",
        "run_handler_util.cc",
        "tensor_slice.cc",
        "tensor_util.cc",
        "versions.cc",
    ],
)

# Files needed for core:mobile_srcs_(no|only)_runtime.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "allocator.cc",
        "allocator.h",
        "allocator_registry.cc",
        "allocator_registry.h",
        "bfloat16.cc",
        "bfloat16.h",
        "bounds_check.h",
        "cpu_allocator_impl.cc",
        "kernel_shape_util.cc",
        "kernel_shape_util.h",
        "log_memory.cc",
        "log_memory.h",
        "numeric_op_base.h",
        "numeric_types.h",
        "op_requires.h",
        "ops_util.cc",
        "ops_util.h",
        "register_types.h",
        "resource_handle.cc",
        "resource_handle.h",
        "tensor.cc",
        "tensor.h",
        "tensor_shape.cc",
        "tensor_shape.h",
        "tensor_types.h",
        "tracking_allocator.cc",
        "tracking_allocator.h",
        "type_index.h",
        "type_traits.h",
        "typed_allocator.cc",
        "typed_allocator.h",
        "types.cc",
        "types.h",
        "variant.cc",
        "variant.h",
        "variant_encode_decode.h",
        "variant_op_registry.cc",
        "variant_op_registry.h",
        "variant_tensor_data.cc",
        "variant_tensor_data.h",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "attr_value_util.cc",
        "attr_value_util.h",
        "cancellation.cc",
        "cancellation.h",
        "collective.cc",
        "collective.h",
        "common_shape_fns.cc",
        "common_shape_fns.h",
        "control_flow.h",
        "dataset.cc",
        "dataset.h",
        "dataset_stateful_op_whitelist.h",
        "device_base.cc",
        "device_base.h",
        "function.cc",
        "function.h",
        "function_handle_cache.cc",
        "function_handle_cache.h",
        "graph_def_util.cc",
        "graph_def_util.h",
        "graph_to_functiondef.cc",
        "graph_to_functiondef.h",
        "kernel_def_builder.cc",
        "kernel_def_builder.h",
        "kernel_def_util.cc",
        "kernel_def_util.h",
        "load_library.cc",
        "local_rendezvous.cc",
        "local_rendezvous.h",
        "logging.cc",
        "logging.h",
        "lookup_interface.cc",
        "lookup_interface.h",
        "memory_types.cc",
        "memory_types.h",
        "metrics.cc",
        "metrics.h",
        "model.cc",
        "model.h",
        "node_def_builder.cc",
        "node_def_builder.h",
        "node_def_util.cc",
        "node_def_util.h",
        "node_properties.cc",
        "node_properties.h",
        "numeric_op.h",
        "op.cc",
        "op.h",
        "op_def_builder.cc",
        "op_def_builder.h",
        "op_def_util.cc",
        "op_def_util.h",
        "op_kernel.cc",
        "op_kernel.h",
        "op_segment.cc",
        "op_segment.h",
        "partial_tensor_shape.h",
        "queue_interface.h",
        "reader_interface.h",
        "register_types_traits.h",
        "rendezvous.cc",
        "rendezvous.h",
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
        "stats_aggregator.h",
        "tensor_reference.h",
        "tensor_slice.cc",
        "tensor_slice.h",
        "tensor_util.cc",
        "tensor_util.h",
        "thread_factory.h",
        "versions.cc",
        "versions.h",
    ],
)

filegroup(
    name = "android_test_hdrs",
    srcs = [
        "fake_input.h",
        "shape_inference_testutil.h",
        "tensor_testutil.h",
    ],
)

filegroup(
    name = "android_test_srcs",
    srcs = [
        "fake_input.cc",
        ":android_test_srcs_no_core",
    ],
)

filegroup(
    name = "android_test_srcs_no_core",
    srcs = [
        "shape_inference_testutil.cc",
        "tensor_testutil.cc",
    ],
)

# Individual targets. These should be prefered over tensorflow/core:framework
# whenever possible.

# This is redundant with the "tensorflow/core:framework" target. It's useful for
# applications that want to depend on a minimal subset of TensorFlow (e.g. XLA).
cc_library(
    name = "allocator",
    srcs = [
        "allocator.cc",
        "allocator_registry.h",
        "tracking_allocator.cc",
        "tracking_allocator.h",
    ],
    hdrs = [
        "allocator.h",
    ],
    features = ["parse_headers"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":numeric_types",
        ":type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ] + if_static(
        extra_deps = [
            ":allocator_registry_impl",
            "//tensorflow/core/lib/gtl:inlined_vector",
            "//tensorflow/core/lib/strings:strcat",
            "//tensorflow/core/lib/strings:stringprintf",
            "//tensorflow/core/platform:env",
            "//tensorflow/core/platform:logging",
            "//tensorflow/core/platform:macros",
            "//tensorflow/core/platform:mutex",
            "//tensorflow/core/platform:platform_port",
            "//tensorflow/core/platform:thread_annotations",
            "//tensorflow/core/platform:types",
        ],
        otherwise = [
            "//tensorflow/core:lib",
        ],
    ),
    alwayslink = 1,
)

# This target will be included in libtensorflow_framework.so via the
# framework_internal_impl target.
# All other dependencies on this target need to go through if_static guard,
# as otherwise duplicate registration in the registry will cause crashes.
cc_library(
    name = "allocator_registry_impl",
    srcs = [
        "allocator.h",
        "allocator_registry.cc",
        "allocator_registry.h",
        "cpu_allocator_impl.cc",
        "tracking_allocator.h",
    ],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":numeric_types",
        ":type_traits",
        "//tensorflow/core/lib/gtl:inlined_vector",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/lib/strings:stringprintf",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensor_testutil",
    testonly = 1,
    srcs = ["tensor_testutil.cc"],
    hdrs = ["tensor_testutil.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "shape_inference_testutil",
    testonly = 1,
    srcs = ["shape_inference_testutil.cc"],
    hdrs = ["shape_inference_testutil.h"],
    copts = tf_copts(),
    deps = [
        ":node_def_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "reader_base",
    srcs = ["reader_base.cc"],
    hdrs = ["reader_base.h"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":reader_base_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "op_gen_lib",
    srcs = ["op_gen_lib.cc"],
    hdrs = ["op_gen_lib.h"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":api_def_proto_cc",
        ":attr_value_proto_cc",
        ":op_def_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/util/proto:proto_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fake_input",
    testonly = 1,
    srcs = ["fake_input.cc"],
    hdrs = ["fake_input.h"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":attr_value_proto_cc",
        ":op_def_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "function_testlib",
    testonly = 1,
    srcs = ["function_testlib.cc"],
    hdrs = ["function_testlib.h"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":function_proto_cc",
        ":graph_proto_cc",
        ":node_def_proto_cc",
        ":tensor_testutil",
        ":versions_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "bfloat16",
    srcs = ["bfloat16.cc"],
    hdrs = ["bfloat16.h"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":numeric_types",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:types",
    ],
    alwayslink = 1,
)

cc_library(
    name = "numeric_types",
    hdrs = ["numeric_types.h"],
    visibility = [
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/core:__subpackages__",
    ],
    deps = [
        "//tensorflow/core/lib/bfloat16",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "bounds_check",
    hdrs = ["bounds_check.h"],
    visibility = ["//tensorflow/core:__pkg__"],
    deps = [
        "//tensorflow/core/platform:macros",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "tensor_shape",
    srcs = ["tensor_shape.cc"],
    hdrs = [
        "partial_tensor_shape.h",
        "tensor_shape.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/runtime_fallback:__pkg__",
    ],
    deps = [
        ":bounds_check",
        ":tensor_shape_proto_cc",
        ":types_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/lib/gtl:inlined_vector",
        "//tensorflow/core/lib/strings:str_util",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/util:overflow",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "resource_handle",
    srcs = ["resource_handle.cc"],
    hdrs = ["resource_handle.h"],
    visibility = ["//tensorflow/core:__pkg__"],
    deps = [
        ":resource_handle_proto_cc",
        ":tensor_shape",
        ":types_proto_cc",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/platform:tensor_coding",
        "//tensorflow/core/platform:types",
    ],
    alwayslink = 1,
)

cc_library(
    name = "type_index",
    hdrs = ["type_index.h"],
    visibility = ["//visibility:private"],
    deps = ["//tensorflow/core/platform:types"],
)

cc_library(
    name = "tensor_types",
    hdrs = ["tensor_types.h"],
    visibility = ["//visibility:private"],
    deps = ["//third_party/eigen3"],
)

cc_library(
    name = "type_traits",
    hdrs = ["type_traits.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":numeric_types",
        "//tensorflow/core/platform:types",
    ],
)

tf_cuda_library(
    name = "tensor",
    srcs = [
        "log_memory.cc",
        "tensor.cc",
        "typed_allocator.cc",
        "types.cc",
        "variant.cc",
        "variant_op_registry.cc",
        "variant_tensor_data.cc",
    ],
    hdrs = [
        "log_memory.h",
        "register_types.h",
        "tensor.h",
        "typed_allocator.h",
        "types.h",
        "variant.h",
        "variant_encode_decode.h",
        "variant_op_registry.h",
        "variant_tensor_data.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/util:__pkg__",
    ],
    deps = [
        ":allocation_description_proto_cc",
        ":allocator",
        ":bfloat16",
        ":log_memory_proto_cc",
        ":numeric_types",
        ":resource_handle",
        ":resource_handle_proto_cc",
        ":tensor_description_proto_cc",
        ":tensor_proto_cc",
        ":tensor_shape",
        ":tensor_types",
        ":type_index",
        ":type_traits",
        ":types_proto_cc",
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:refcount",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/lib/gtl:flatmap",
        "//tensorflow/core/lib/gtl:inlined_vector",
        "//tensorflow/core/lib/hash",
        "//tensorflow/core/lib/strings:str_util",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/platform:abi",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:tensor_coding",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/public:version",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
        ":bounds_check",
        ":node_def_proto_cc",
        ":node_def_util",
        ":tensor",
        ":tensor_shape",
        ":tensor_shape_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/gtl:inlined_vector",
        "//tensorflow/core/lib/strings:numbers",
        "//tensorflow/core/lib/strings:scanner",
        "//tensorflow/core/lib/strings:str_util",
        "//tensorflow/core/platform:macros",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "kernel_shape_util",
    srcs = ["kernel_shape_util.cc"],
    hdrs = ["kernel_shape_util.h"],
    deps = [
        ":tensor",
        ":tensor_shape",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/util:padding",
    ],
)

cc_library(
    name = "common_shape_fns",
    srcs = ["common_shape_fns.cc"],
    hdrs = ["common_shape_fns.h"],
    deps = [
        ":attr_value_proto_cc",
        ":shape_inference",
        ":tensor",
        ":tensor_shape",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/gtl:inlined_vector",
        "//tensorflow/core/util:einsum_op_util",
        "//tensorflow/core/util:padding",
        "//tensorflow/core/util:tensor_format",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "attr_value_util",
    srcs = ["attr_value_util.cc"],
    hdrs = ["attr_value_util.h"],
    deps = [
        ":attr_value_proto_text",
        ":tensor",
        ":tensor_shape",
        ":tensor_shape_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/lib/hash",
        "//tensorflow/core/lib/strings:proto_serialization",
        "//tensorflow/core/lib/strings:str_util",
        "//tensorflow/core/platform:protobuf",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_def_util",
    srcs = ["op_def_util.cc"],
    hdrs = ["op_def_util.h"],
    deps = [
        ":api_def_proto_cc",
        ":attr_value_proto_cc",
        ":attr_value_util",
        ":op_def_proto_cc",
        ":tensor",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/gtl:map_util",
        "//tensorflow/core/lib/hash",
        "//tensorflow/core/lib/strings:proto_serialization",
        "//tensorflow/core/lib/strings:scanner",
        "//tensorflow/core/lib/strings:str_util",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "node_def_util",
    srcs = ["node_def_util.cc"],
    hdrs = ["node_def_util.h"],
    deps = [
        ":attr_value_proto_cc",
        ":attr_value_util",
        ":node_def_proto_cc",
        ":op_def_proto_cc",
        ":op_def_util",
        ":tensor",
        ":tensor_proto_cc",
        ":tensor_shape",
        ":tensor_shape_proto_cc",
        ":types_proto_cc",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/lib/gtl:flatmap",
        "//tensorflow/core/lib/gtl:map_util",
        "//tensorflow/core/lib/hash",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:scanner",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/util:padding",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "node_properties",
    srcs = ["node_properties.cc"],
    hdrs = ["node_properties.h"],
    deps = [
        ":node_def_proto_cc",
        ":node_def_util",
        ":op",
        ":op_def_proto_cc",
        ":tensor",
        ":types_proto_cc",
        "//tensorflow/core/lib/core:status",
    ],
)

cc_library(
    name = "op_def_builder",
    srcs = ["op_def_builder.cc"],
    hdrs = ["op_def_builder.h"],
    deps = [
        ":attr_value_proto_cc",
        ":attr_value_util",
        ":op_def_proto_cc",
        ":op_def_util",
        ":tensor",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/lib/strings:scanner",
        "//tensorflow/core/lib/strings:str_util",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/platform:macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "selective_registration",
    hdrs = ["selective_registration.h"],
)

cc_library(
    name = "op",
    srcs = ["op.cc"],
    hdrs = ["op.h"],
    deps = [
        ":op_def_builder",
        ":op_def_util",
        ":selective_registration",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/gtl:map_util",
        "//tensorflow/core/lib/strings:str_util",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "op_requires",
    hdrs = ["op_requires.h"],
    deps = ["//tensorflow/core/platform:macros"],
)

# Files whose users still need to be migrated from core:framework to the
# above targets.
# TODO(gonnet): Remove these files once targets depending on them have
# been cleaned up.
exports_files(
    srcs = [
        "allocator.h",
        "bfloat16.h",
        "bounds_check.h",
        "fake_input.h",
        "function_testlib.h",
        "log_memory.h",
        "numeric_types.h",
        "op_gen_lib.h",
        "reader_base.h",
        "register_types.h",
        "resource_handle.h",
        "shape_inference_testutil.h",
        "tensor.h",
        "tensor_interface.h",
        "tensor_shape.h",
        "tensor_testutil.h",
        "tensor_types.h",
        "type_index.h",
        "type_traits.h",
        "typed_allocator.h",
        "types.h",
        "variant.h",
        "variant_encode_decode.h",
        "variant_op_registry.h",
        "variant_tensor_data.h",
    ],
)

# Framework tests.
tf_cc_tests(
    name = "higher_level_tests",
    size = "small",
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
        "node_properties_test.cc",
        "op_compatibility_test.cc",
        "op_def_builder_test.cc",
        "op_def_util_test.cc",
        "op_kernel_test.cc",
        "op_registration_test.cc",
        "partial_tensor_shape_test.cc",
        "rendezvous_test.cc",
        "resource_mgr_test.cc",
        "resource_op_kernel_test.cc",
        "shape_inference_test.cc",
        "shape_inference_testutil_test.cc",
        "tensor_shape_test.cc",
        "tensor_slice_test.cc",
        "tensor_test.cc",
        "tensor_testutil_test.cc",
        "tensor_util_test.cc",
        "tracking_allocator_test.cc",
        "types_test.cc",
        "variant_op_registry_test.cc",
        "variant_test.cc",
    ],
    create_named_test_suite = True,
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    linkstatic = tf_kernel_tests_linkstatic(),
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/core:__pkg__",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/cc:while_loop",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/util:protos_test_cc",
        "//third_party/eigen3",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "op_gen_lib.h",
        "rendezvous.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
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
