load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
    "tf_pyclif_proto_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_generate_proto_text_sources",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
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
        "attr_value_util.h",
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
        "log_memory.h",
        "logging.h",
        "lookup_interface.h",
        "memory_types.h",
        "model.h",
        "node_def_builder.h",
        "node_def_util.h",
        "numeric_op.h",
        "op.h",
        "op_def_builder.h",
        "op_def_util.h",
        "op_kernel.h",
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
        "unique_tensor_references.h",
        "variant.h",
        "variant_encode_decode.h",
        "variant_op_registry.h",
        "variant_tensor_data.h",
        "versions.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
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
            "allocator_registry.cc",
            "bfloat16.cc",
            "cpu_allocator_impl.cc",
            "fake_input.*",
            "op_gen_lib.*",
            "reader_base.*",
            "tracking_allocator.cc",
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

filegroup(
    name = "android_test_hdrs",
    srcs = [
        ":fake_input.h",
        ":shape_inference_testutil.h",
        ":tensor_testutil.h",
    ],
)

filegroup(
    name = "android_test_srcs",
    srcs = [
        ":android_test_srcs_no_core",
        ":fake_input.cc",
    ],
)

filegroup(
    name = "android_test_srcs_no_core",
    srcs = [
        ":shape_inference_testutil.cc",
        ":tensor_testutil.cc",
    ],
)

filegroup(
    name = "test_srcs",
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
        "op_kernel_test.cc",
        "op_registration_test.cc",
        "partial_tensor_shape_test.cc",
        "rendezvous_test.cc",
        "resource_mgr_test.cc",
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
        "variant_op_registry_test.cc",
        "variant_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

# Put this test in its own filegroup so that it is compiled and run separately
# since it relies on global state, i.e. the ResourceManager being untouched.
filegroup(
    name = "resource_op_kernel_test_srcs",
    srcs = ["resource_op_kernel_test.cc"],
    visibility = ["//tensorflow/core:__pkg__"],
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
        "numeric_types.h",
        "tracking_allocator.cc",
        "tracking_allocator.h",
        "type_traits.h",
    ],
    hdrs = [
        "allocator.h",
    ],
    features = ["parse_headers"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//third_party/eigen3",
        "//tensorflow/core:lib",
    ] + if_static(extra_deps = [":allocator_registry_impl"]),
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
        "numeric_types.h",
        "tracking_allocator.h",
        "type_traits.h",
    ],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        "//tensorflow/core:lib",
        "//third_party/eigen3",
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
)

cc_library(
    name = "numeric_types",
    hdrs = ["numeric_types.h"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        "//tensorflow/core/lib/bfloat16",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
)

# Files whose users still need to be migrated from core:framework to the
# above targets.
# TODO(gonnet): Remove these files once targets depending on them have
# been cleaned up.
exports_files(
    srcs = [
        "allocator.h",
        "bfloat16.h",
        "fake_input.h",
        "function_testlib.h",
        "numeric_types.h",
        "op_gen_lib.h",
        "reader_base.h",
        "shape_inference_testutil.h",
        "tensor_testutil.h",
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
