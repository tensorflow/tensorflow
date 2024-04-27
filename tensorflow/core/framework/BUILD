# Placeholder: load py_proto_library
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_copts",
    "tf_cuda_library",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_cuda_cc_test", "tf_generate_proto_text_sources")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
    "tf_pyclif_proto_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

default_visibility = [
    "//tensorflow/core:__subpackages__",
    "//tensorflow/security/fuzzing:__subpackages__",
    # TODO(pedaveeraiah): to be removed when summary.proto.h deps moves to TSL
    "@local_tsl//tsl/lib:__subpackages__",
    # copybara:uncomment "//learning/brain/tfrt/aot:__subpackages__",
    # copybara:uncomment "//platforms/xla/megascale/tensorflow:__subpackages__",
    # copybara:uncomment "//learning/brain/experimental/tfrt/native_lowering/graph_executor:__subpackages__",
    "//tensorflow/cc/saved_model:__subpackages__",
    #internal starburst clustering service,
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

# Export all header files for which we do not yet provide a dedicated build
# rule. This avoids breaking all the rules in tensorflow/core/BUILD.
exports_files(
    srcs = [
        "allocator_registry.h",
        "collective.h",
        "control_flow.h",
        "dataset.h",
        "dataset_stateful_op_allowlist.h",
        "device.h",
        "device_base.h",
        "device_factory.h",
        "function.h",
        "function_handle_cache.h",
        "graph_def_util.h",
        "graph_to_functiondef.h",
        "kernel_def_builder.h",
        "kernel_def_util.h",
        "local_rendezvous.h",
        "logging.h",
        "lookup_interface.h",
        "memory_types.h",
        "metrics.h",
        "model.h",
        "node_def_builder.h",
        "numeric_op.h",
        "op_kernel.h",
        "op_requires.h",
        "op_segment.h",
        "ops_util.h",
        "partial_tensor_shape.h",
        "queue_interface.h",
        "reader_interface.h",
        "reader_op_kernel.h",
        "ref_var.h",
        "register_types_traits.h",
        "rendezvous.h",
        "resource_mgr.h",
        "resource_op_kernel.h",
        "resource_var.h",
        "rng_alg.h",
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
        "cpp_shape_inference.proto",
        "dataset_metadata.proto",
        "dataset_options.proto",
        "device_attributes.proto",
        "full_type.proto",
        "function.proto",
        "graph.proto",
        "graph_debug_info.proto",
        "graph_transfer_info.proto",
        "kernel_def.proto",
        "log_memory.proto",
        "model.proto",
        "node_def.proto",
        "op_def.proto",
        "reader_base.proto",
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
        "full_type_util.h",
        "full_type_inference_util.h",
        "op_def_util.h",
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
        "dataset_stateful_op_allowlist.h",
        "device.h",
        "device_base.h",
        "device_factory.h",
        "full_type_inference_util.h",
        "full_type_util.h",
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
        "ref_var.h",
        "register_types.h",
        "register_types_traits.h",
        "rendezvous.h",
        "resource_base.h",
        "resource_handle.h",
        "resource_mgr.h",
        "resource_op_kernel.h",
        "resource_var.h",
        "run_handler.h",
        "run_handler_util.h",
        "session_state.h",
        "shape_inference.h",
        "shared_ptr_variant.h",
        "stats_aggregator.h",
        "tensor.h",
        "tensor_key.h",
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
        "//tensorflow/core/framework/registration:options.h",
        "//tensorflow/core/framework/registration:registration.h",
        "@local_tsl//tsl/framework:device_type.h",
    ],
)

filegroup(
    name = "framework_internal_impl_srcs",
    srcs = [
        "collective.cc",
        "dataset.cc",
        "device.cc",
        "device_base.cc",
        "device_factory.cc",
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
        "ref_var.cc",
        "rendezvous.cc",
        "resource_mgr.cc",
        "resource_var.cc",
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
        "allocator.h",
        "allocator_registry.h",
        "bfloat16.cc",
        "bfloat16.h",
        "bounds_check.h",
        "kernel_shape_util.cc",
        "kernel_shape_util.h",
        "log_memory.cc",
        "log_memory.h",
        "numeric_types.h",
        "op_requires.h",
        "ops_util.cc",
        "ops_util.h",
        "register_types.h",
        "resource_base.h",
        "resource_handle.cc",
        "resource_handle.h",
        "tensor.cc",
        "tensor.h",
        "tensor_key.h",
        "tensor_shape.cc",
        "tensor_shape.h",
        "tensor_types.h",
        "tensor_util.h",
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
        "@local_tsl//tsl/framework:device_type.h",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "attr_value_util.cc",
        "attr_value_util.h",
        "cancellation.h",
        "collective.cc",
        "collective.h",
        "common_shape_fns.cc",
        "common_shape_fns.h",
        "control_flow.h",
        "dataset.cc",
        "dataset.h",
        "dataset_stateful_op_allowlist.h",
        "device.cc",
        "device.h",
        "device_base.cc",
        "device_base.h",
        "device_factory.cc",
        "device_factory.h",
        "full_type_inference_util.cc",
        "full_type_inference_util.h",
        "full_type_util.cc",
        "full_type_util.h",
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
        "reader_base.cc",
        "reader_base.h",
        "reader_interface.h",
        "reader_op_kernel.h",
        "ref_var.cc",
        "ref_var.h",
        "register_types_traits.h",
        "rendezvous.cc",
        "rendezvous.h",
        "resource_mgr.cc",
        "resource_mgr.h",
        "resource_op_kernel.h",
        "resource_var.cc",
        "resource_var.h",
        "rng_alg.h",
        "run_handler.cc",
        "run_handler.h",
        "run_handler_util.cc",
        "run_handler_util.h",
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
        "//tensorflow/core/framework/registration:options.h",
        "//tensorflow/core/framework/registration:registration.h",
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

# Individual targets. These should be preferred over tensorflow/core:framework
# whenever possible.

# This is redundant with the "tensorflow/core:framework" target. It's useful for
# applications that want to depend on a minimal subset of TensorFlow (e.g. XLA).
cc_library(
    name = "allocator",
    hdrs = [
        "allocator.h",
    ],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":numeric_types",
        ":type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/framework:allocator",
    ] + if_static(
        extra_deps = [
            ":allocator_registry_impl",
            "//tensorflow/core/lib/gtl:inlined_vector",
            "//tensorflow/core/lib/strings:strcat",
            "//tensorflow/core/lib/strings:stringprintf",
            "//tensorflow/core/platform:env",
            "//tensorflow/core/platform:env_impl",
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
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        "@local_tsl//tsl/framework:allocator_registry_impl",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensor_testutil",
    testonly = 1,
    srcs = ["tensor_testutil.cc"],
    hdrs = ["tensor_testutil.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "tensor_matcher",
    testonly = True,
    srcs = ["tensor_matcher.cc"],
    hdrs = ["tensor_matcher.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "shape_inference_testutil",
    testonly = 1,
    srcs = ["shape_inference_testutil.cc"],
    hdrs = ["shape_inference_testutil.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:internal"],
    deps = [
        ":node_def_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reader_base",
    srcs = ["reader_base.cc"],
    hdrs = ["reader_base.h"],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//tensorflow:internal"],
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
    visibility = [
        "//tensorflow/core:__subpackages__",
        "//tensorflow/security/fuzzing:__subpackages__",
    ],
    deps = [
        ":numeric_types",
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:types",
        "@eigen_archive//:eigen3",
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
        "//tensorflow/core/platform:types",
        "@local_tsl//tsl/framework:numeric_types",
    ],
)

cc_library(
    name = "bounds_check",
    hdrs = ["bounds_check.h"],
    visibility = ["//tensorflow/core/kernels:friends"],
    deps = [
        "//tensorflow/core/platform:macros",
        "@eigen_archive//:eigen3",
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
        "//tensorflow/core/runtime_fallback:__subpackages__",
        "//tensorflow/core/tfrt/utils:__subpackages__",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/util:overflow",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "resource_base",
    hdrs = ["resource_base.h"],
    visibility = default_visibility + [
        "//learning/brain/google/data/core/kernels:__pkg__",
        "//learning/deepmind/tensorflow/queues:__pkg__",
        "//learning/deepmind/tensorflow/sstable:__pkg__",
        "//tensorflow/compiler/mlir/tools/kernel_gen:__pkg__",
        "//waymo/ml/compiler/frontend/kernels:__pkg__",
    ],
    deps = [
        "//tensorflow/core/lib/core:refcount",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "resource_handle",
    srcs = ["resource_handle.cc"],
    hdrs = ["resource_handle.h"],
    visibility = [
        "//tensorflow/compiler/mlir/tensorflow:__subpackages__",
        "//tensorflow/core:__pkg__",
    ],
    deps = [
        ":resource_base",
        ":resource_handle_proto_cc",
        ":tensor_shape",
        ":type_index",
        ":types_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/strings:strcat",
        "//tensorflow/core/platform:casts",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:intrusive_ptr",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:tensor_coding",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/util:managed_stack_trace",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "resource_handle_test",
    size = "small",
    srcs = ["resource_handle_test.cc"],
    deps = [
        ":resource_handle",
        ":resource_handle_proto_cc",
        ":tensor_shape_proto_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "type_index",
    hdrs = ["type_index.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "tensor_types",
    hdrs = ["tensor_types.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core/platform:logging",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "type_traits",
    hdrs = ["type_traits.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":numeric_types",
        "//tensorflow/core/platform:types",
        "@local_tsl//tsl/framework:type_traits",
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
        "tensor_util.h",
        "typed_allocator.h",
        "types.h",
        "variant.h",
        "variant_encode_decode.h",
        "variant_op_registry.h",
        "variant_tensor_data.h",
    ],
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/cc/saved_model:__subpackages__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/runtime_fallback:__subpackages__",
        "//tensorflow/core/tfrt:__subpackages__",
        "//tensorflow/core/util:__pkg__",
        "//tensorflow/security/fuzzing:__subpackages__",
    ],
    deps = [
        ":allocation_description_proto_cc",
        ":allocator",
        ":bfloat16",
        ":full_type_proto_cc",
        ":log_memory_proto_cc",
        ":numeric_types",
        ":resource_handle",
        ":resource_handle_proto_cc",
        ":tensor_description_proto_cc",
        ":tensor_proto_cc",
        ":tensor_shape",
        ":tensor_shape_proto_cc",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:tensor_coding",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/public:version",
        "//tensorflow/core/util:managed_stack_trace",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/framework:device_type",
        "@local_xla//xla/tsl/util:byte_swap_array",
    ],
    alwayslink = 1,
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    deps = [
        ":bounds_check",
        ":full_type_proto_cc",
        ":full_type_util",
        ":node_def_util",
        ":op_def_proto_cc",
        ":tensor_shape",
        ":tensor_shape_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/util:overflow",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
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
        "//tensorflow/core/platform:fingerprint",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/util:overflow",
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
        "//tensorflow/core/lib/strings:proto_serialization",
        "//tensorflow/core/lib/strings:scanner",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
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
        ":full_type_proto_cc",
        ":node_def_proto_cc",
        ":node_def_util",
        ":op",
        ":op_def_builder",
        ":op_def_proto_cc",
        ":tensor",
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
        ":full_type_proto_cc",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "@com_google_absl//absl/strings",
    ],
)

# TODO(mdan): Move these into a separate directory.
cc_library(
    name = "full_type_util",
    srcs = [
        "full_type_inference_util.cc",
        "full_type_util.cc",
    ],
    hdrs = [
        "full_type_inference_util.h",
        "full_type_util.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":attr_value_proto_cc",
        ":full_type_proto_cc",
        ":node_def_proto_cc",
        ":node_def_util",
        ":op_def_builder",
        ":op_def_proto_cc",
        ":tensor",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op",
    srcs = ["op.cc"],
    hdrs = ["op.h"],
    deps = [
        ":full_type_proto_cc",
        ":full_type_util",
        ":loose_headers",
        ":op_def_builder",
        ":op_def_util",
        "//tensorflow/core:core_stringpiece",
        "//tensorflow/core/framework/registration",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/gtl:map_util",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
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
        "cancellation.h",
        "fake_input.h",
        "function_testlib.h",
        "log_memory.h",
        "numeric_types.h",
        "op_gen_lib.h",
        "reader_base.h",
        "register_types.h",
        "resource_base.h",
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
tf_cc_test(
    name = "framework_op_gen_lib_test",
    size = "small",
    srcs = ["op_gen_lib_test.cc"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:op_gen_lib",
    ],
)

tf_cuda_cc_test(
    name = "variant_op_copy_test",
    size = "small",
    srcs = ["variant_op_copy_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "framework_run_handler_util_test",
    size = "small",
    srcs = ["run_handler_util_test.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "framework_run_handler_test",
    size = "small",
    srcs = ["run_handler_test.cc"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:matmul_op",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "framework_op_segment_test",
    size = "small",
    srcs = ["op_segment_test.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
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
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:ops_util",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "framework_resource_var_test",
    size = "small",
    srcs = ["resource_var_test.cc"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "op_kernel_test_base",
    testonly = True,
    hdrs = [
        "op_kernel_test_base.h",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_tests(
    name = "higher_level_tests",
    size = "small",
    srcs = [
        "allocator_test.cc",
        "attr_value_util_test.cc",
        "batch_util_test.cc",
        "bfloat16_test.cc",
        "common_shape_fns_test.cc",
        "dataset_test.cc",
        "device_base_test.cc",
        "disable_jit_test.cc",
        "full_type_inference_util_test.cc",
        "full_type_util_test.cc",
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
        "tensor_matcher_test.cc",
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
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/core:__pkg__",
    ],
    deps = [
        ":op_kernel_test_base",
        ":tensor_matcher",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/cc:while_loop",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/protobuf:memory_profile_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "//tensorflow/core/util:protos_test_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "pywrap_required_hdrs",
    textual_hdrs = [
        "op_gen_lib.h",
        "rendezvous.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/client:__pkg__",
        "//tensorflow/python/util:__pkg__",
    ],
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed. See
# go/loose-lsc-one-target-approach for more details.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = ["op_gen_lib.h"],
    visibility = ["//visibility:private"],
)

# All framework protos are self-contained, i.e. they only import other
# protos from the same package, so we can build the protos here and then
# link them from core:protos_all without circular dependencies.

# Generate the C++ sources for some of the protos.
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
        ":graph_debug_info_proto",
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
        ":full_type_proto",
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "graph_debug_info_proto",
    srcs = ["graph_debug_info.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
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
        ":full_type_proto",
        ":resource_handle_proto",
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
    visibility = [
        "//tensorflow/core:__subpackages__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/security/fuzzing:__subpackages__",
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
    visibility = [
        "//tensorflow/core:__subpackages__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/security/fuzzing:__subpackages__",
    ],
)

tf_proto_library(
    name = "full_type_proto",
    srcs = ["full_type.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [],
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
    name = "cpp_shape_inference_proto",
    srcs = ["cpp_shape_inference.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":full_type_proto",
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

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "function_proto_py_pb2",
#     has_services = 0,
#     api_version = 2,
#     deps = [
#         ":function_proto",
#     ],
# )
# copybara:uncomment_end

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
        "@local_tsl//tsl/protobuf:histogram_proto",
    ],
    exports = ["@local_tsl//tsl/protobuf:histogram_proto"],
)

tf_proto_library(
    name = "model_proto",
    srcs = ["model.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "dataset_proto",
    srcs = ["dataset.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":tensor_proto",
        ":tensor_shape_proto",
        ":types_proto",
    ],
)

tf_proto_library(
    name = "dataset_metadata_proto",
    srcs = ["dataset_metadata.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "dataset_options_proto",
    srcs = ["dataset_options.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":model_proto",
    ],
)

tf_proto_library(
    name = "optimized_function_graph_proto",
    srcs = ["optimized_function_graph.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":types_proto",
        ":graph_proto",
    ],
)

tf_proto_library(
    name = "protos_all",
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        ":allocation_description_proto",
        ":api_def_proto",
        ":cpp_shape_inference_proto",
        ":attr_value_proto",
        ":cost_graph_proto",
        ":dataset_proto",
        ":dataset_metadata_proto",
        ":dataset_options_proto",
        ":device_attributes_proto",
        ":full_type_proto",
        ":function_proto",
        ":graph_debug_info_proto",
        ":graph_proto",
        ":graph_transfer_info_proto",
        ":kernel_def_proto",
        ":log_memory_proto",
        ":model_proto",
        ":node_def_proto",
        ":optimized_function_graph_proto",
        ":op_def_proto",
        ":reader_base_proto",
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
    tags = [
        "alt_dep=//third_party/tensorflow/core:protos_all",
    ],
)

tf_cc_fuzz_test(
    name = "tensor_shape_fuzz",
    srcs = ["tensor_shape_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_shape_domains",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

tf_cc_fuzz_test(
    name = "tensor_fuzz",
    srcs = ["tensor_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/security/fuzzing/cc/core/framework:datatype_domains",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_domains",
        "//tensorflow/security/fuzzing/cc/core/framework:tensor_shape_domains",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

tf_cc_test(
    name = "metrics_test",
    srcs = ["metrics_test.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "@com_google_googletest//:gtest_main",
    ],
)
