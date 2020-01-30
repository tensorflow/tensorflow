# Description:
#   Base testing infrastructure for XLA.

load("//tensorflow/compiler/xla/tests:build_defs.bzl", "generate_backend_suites", "generate_backend_test_macros", "xla_test", "xla_test_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
    packages = ["//platforms/testing/tests/..."],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

# Generate test_suites for all backends, named "${backend}_tests".
generate_backend_suites()

# Target to add main for tests. Do not link this target and
# //third_party/tensorflow/core:test_main into the same target.
cc_library(
    name = "xla_internal_test_main",
    testonly = True,
    srcs = ["xla_internal_test_main.cc"],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

cc_library(
    name = "test_macros_header",
    testonly = True,
    hdrs = ["test_macros.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

# Generate a test_macros_${BACKEND} library per backend with the proper copts.
generate_backend_test_macros()

cc_library(
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_casting_utils",
        "//tensorflow/compiler/xla/service:hlo_dataflow_analysis",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "literal_test_util",
    testonly = True,
    srcs = ["literal_test_util.cc"],
    hdrs = ["literal_test_util.h"],
    deps = [
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_comparison",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "verified_hlo_module",
    testonly = True,
    srcs = ["verified_hlo_module.cc"],
    hdrs = ["verified_hlo_module.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_test_base",
    testonly = True,
    srcs = ["hlo_test_base.cc"],
    hdrs = ["hlo_test_base.h"],
    deps = [
        ":literal_test_util",
        ":test_utils",
        ":verified_hlo_module",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/service:interpreter_plugin",  # reference backend
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_binary(
    name = "local_client_aot_test_helper",
    srcs = ["local_client_aot_test_helper.cc"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:support",
    ],
)

genrule(
    name = "local_client_aot_test_computation",
    outs = ["local_client_aot_test_computation.o"],
    cmd = "$(location :local_client_aot_test_helper) $(TARGET_CPU) > $(OUTS)",
    local = 1,
    tools = [":local_client_aot_test_helper"],
)

cc_library(
    name = "client_library_test_base",
    testonly = True,
    srcs = ["client_library_test_base.cc"],
    hdrs = ["client_library_test_base.h"],
    deps = [
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:interpreter_plugin",  # reference backend
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "llvm_irgen_test_base",
    testonly = True,
    srcs = ["llvm_irgen_test_base.cc"],
    hdrs = ["llvm_irgen_test_base.h"],
    deps = [
        ":codegen_test_base",
        ":filecheck",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:test",
    ],
)

cc_library(
    name = "codegen_test_base",
    testonly = True,
    srcs = ["codegen_test_base.cc"],
    hdrs = ["codegen_test_base.h"],
    deps = [
        ":hlo_test_base",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
    ],
)

cc_library(
    name = "filecheck",
    testonly = True,
    srcs = ["filecheck.cc"],
    hdrs = ["filecheck.h"],
    data = [
        "@llvm-project//llvm:FileCheck",
    ],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "local_client_test_base",
    testonly = True,
    srcs = ["local_client_test_base.cc"],
    hdrs = ["local_client_test_base.h"],
    deps = [
        ":client_library_test_base",
        ":verified_hlo_module",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/stream_executor:device_memory_allocator",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "bad_rng_shape_validation_test",
    srcs = ["bad_rng_shape_validation_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "buffer_donation_test",
    srcs = ["buffer_donation_test.cc"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

xla_test(
    name = "conv_depthwise_test",
    timeout = "long",
    srcs = [
        "conv_depthwise_test.cc",
    ],
    shard_count = 50,
    deps = [
        ":conv_depthwise_common",
        ":test_macros_header",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:bfloat16_normalization",
        "//tensorflow/compiler/xla/service:despecializer",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:optional",
    ],
)

xla_test(
    name = "conv_depthwise_backprop_filter_test",
    timeout = "long",
    srcs = ["conv_depthwise_backprop_filter_test.cc"],
    # these backends do not natively handle batch group counts.
    blacklisted_backends = [
        "gpu",
        "cpu",
    ],
    shard_count = 6,
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:bfloat16_normalization",
        "//tensorflow/compiler/xla/service:despecializer",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:optional",
    ],
)

xla_test(
    name = "grouped_convolution_test",
    timeout = "long",
    srcs = ["grouped_convolution_test.cc"],
    blacklisted_backends = [
        # disabled because it times out.
        "cpu",
    ],
    shard_count = 50,
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:bfloat16_normalization",
        "//tensorflow/compiler/xla/service:despecializer",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:optional",
    ],
)

xla_test(
    name = "check_execution_arity_test",
    srcs = ["check_execution_arity_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "query_inferred_shape_test",
    srcs = ["query_inferred_shape_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "while_test",
    srcs = ["while_test.cc"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "xla_hlo_profile_test",
    srcs = ["xla_hlo_profile_test.cc"],
    blacklisted_backends = [
        # Hlo profiles are not supported on the interpreter backend.
        "interpreter",
    ],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":test_utils",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:stream_pool",
        "//tensorflow/core:lib",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "axpy_simple_test",
    srcs = ["axpy_simple_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "map_test",
    srcs = ["map_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "params_test",
    timeout = "long",
    srcs = ["params_test.cc"],
    shard_count = 30,
    tags = [
        "optonly",
    ],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "pred_test",
    srcs = ["pred_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "select_test",
    srcs = ["select_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "conditional_test",
    srcs = ["conditional_test.cc"],
    shard_count = 2,
    tags = ["no_rocm"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "unary_op_test",
    srcs = ["unary_op_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "scalar_computations_test",
    srcs = ["scalar_computations_test.cc"],
    shard_count = 32,
    tags = ["no_rocm"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "deallocation_test",
    srcs = ["deallocation_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "deconstruct_tuple_test",
    srcs = ["deconstruct_tuple_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "array_elementwise_ops_test",
    srcs = ["array_elementwise_ops_test.cc"],
    shard_count = 25,
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "exhaustive_op_test_utils",
    testonly = True,
    srcs = ["exhaustive_op_test_utils.cc"],
    hdrs = [
        "exhaustive_op_test_utils.h",
    ],
    tags = ["no_pip"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:bit_cast",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/client/lib:math",
    ],
)

cc_library(
    name = "conv_depthwise_common",
    testonly = True,
    srcs = ["conv_depthwise_common.cc"],
    hdrs = ["conv_depthwise_common.h"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:bfloat16_normalization",
        "//tensorflow/compiler/xla/service:despecializer",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/types:optional",
    ],
)

xla_test(
    name = "exhaustive_unary_test_f32_or_smaller",
    srcs = ["exhaustive_unary_test.cc"],
    copts = ["-DUNARY_TEST_TARGET_F32_OR_SMALLER"],
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":client_library_test_base",
        ":exhaustive_op_test_utils",
        "//tensorflow/compiler/xla:util",
    ],
)

xla_test(
    name = "exhaustive_unary_test_f64",
    srcs = ["exhaustive_unary_test.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    copts = ["-DUNARY_TEST_TARGET_F64"],
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":client_library_test_base",
        ":exhaustive_op_test_utils",
        "//tensorflow/compiler/xla:util",
    ],
)

xla_test(
    name = "exhaustive_unary_test_complex",
    srcs = ["exhaustive_unary_test.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    copts = ["-DUNARY_TEST_TARGET_COMPLEX"],
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":client_library_test_base",
        ":exhaustive_op_test_utils",
        "//tensorflow/compiler/xla:util",
    ],
)

xla_test(
    name = "exhaustive_binary_test_f16",
    srcs = ["exhaustive_binary_test.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    copts = ["-DBINARY_TEST_TARGET_F16"],
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":exhaustive_op_test_utils",
    ],
)

xla_test(
    name = "exhaustive_binary_test_bf16",
    srcs = ["exhaustive_binary_test.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    copts = ["-DBINARY_TEST_TARGET_BF16"],
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":exhaustive_op_test_utils",
    ],
)

xla_test(
    name = "exhaustive_binary_test_f32",
    srcs = ["exhaustive_binary_test.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    copts = ["-DBINARY_TEST_TARGET_F32"],
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":exhaustive_op_test_utils",
    ],
)

xla_test(
    name = "exhaustive_binary_test_f64",
    srcs = ["exhaustive_binary_test.cc"],
    backends = [
        "gpu",
        "cpu",
    ],
    copts = ["-DBINARY_TEST_TARGET_F64"],
    real_hardware_only = True,  # Very slow on the interpreter.
    shard_count = 48,
    tags = [
        "optonly",
        # This is a big test that we skip for capacity reasons in OSS testing.
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":exhaustive_op_test_utils",
    ],
)

xla_test(
    name = "reduce_precision_test",
    srcs = ["reduce_precision_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "dot_operation_test",
    srcs = ["dot_operation_test.cc"],
    shard_count = 20,
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

# Run dot tests with auto-tuning disabled.  This just does a basic sanity check
# that setting xla_gpu_autotune_level to 0 does not break simple graphs.
xla_test(
    name = "dot_operation_test_autotune_disabled",
    srcs = ["dot_operation_test.cc"],
    args = ["--xla_gpu_autotune_level=0"],
    backends = ["gpu"],
    shard_count = 20,
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "gather_operation_test",
    srcs = ["gather_operation_test.cc"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "scatter_test",
    srcs = ["scatter_test.cc"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":test_macros_header",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

# Repeat dot_operation_runtime_test with single-threaded eigen.
xla_test(
    name = "dot_operation_single_threaded_runtime_test",
    srcs = ["dot_operation_test.cc"],
    backend_args = {
        "cpu": [
            "--xla_cpu_multi_thread_eigen=false",
        ],
    },
    shard_count = 20,
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "constants_test",
    srcs = ["constants_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

CONVOLUTION_TEST_DEPS = [
    ":test_macros_header",
    "//tensorflow/compiler/xla:array2d",
    "//tensorflow/compiler/xla:array4d",
    "//tensorflow/compiler/xla:literal",
    "//tensorflow/compiler/xla:reference_util",
    "//tensorflow/compiler/xla:shape_util",
    "//tensorflow/compiler/xla:statusor",
    "//tensorflow/compiler/xla:util",
    "//tensorflow/compiler/xla:xla_data_proto_cc",
    "//tensorflow/compiler/xla/client:global_data",
    "//tensorflow/compiler/xla/client:local_client",
    "//tensorflow/compiler/xla/client:padding",
    "//tensorflow/compiler/xla/client:xla_builder",
    "//tensorflow/compiler/xla/tests:client_library_test_base",
    "//tensorflow/compiler/xla/tests:hlo_test_base",
    "//tensorflow/compiler/xla/tests:literal_test_util",
    "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    "//tensorflow/core:lib",
    "//tensorflow/core:test",
]

xla_test(
    name = "convolution_test",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    shard_count = 40,
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

# Run convolution tests with auto-tuning disabled.  This just does a basic
# sanity check that setting xla_gpu_autotune_level to 0 does not break simple
# graphs.
xla_test(
    name = "convolution_test_autotune_disabled",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    args = ["--xla_gpu_autotune_level=0"],
    backends = ["gpu"],
    shard_count = 40,
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "convolution_test_gpu_alternative_layout",
    timeout = "long",
    srcs = ["convolution_test.cc"],
    backend_args = {"gpu": ["--xla_backend_extra_options=xla_gpu_experimental_conv_disable_layout_heuristic"]},
    backends = ["gpu"],
    shard_count = 25,
    tags = ["no_rocm"],
    deps = CONVOLUTION_TEST_DEPS + [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "convolution_variants_test",
    timeout = "long",
    srcs = ["convolution_variants_test.cc"],
    backend_tags = {
        # TODO(b/31436974): Fix msan failure. Failed on 2016-09-12.
        "cpu": ["nomsan"],
    },
    shard_count = 30,
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "convolution_dimension_numbers_test",
    timeout = "long",
    srcs = ["convolution_dimension_numbers_test.cc"],
    shard_count = 20,
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

xla_test(
    name = "batch_normalization_test",
    srcs = ["batch_normalization_test.cc"],
    blacklisted_backends = [
        # BatchNorm HLOs are not handled by the interpreter backend, and the
        # BatchNorm expander is not run on the interpreter.
        "interpreter",
    ],
    shard_count = 40,
    tags = ["no_rocm"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core/lib/math:math_util",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:test",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "bfloat16_test",
    srcs = ["bfloat16_test.cc"],
    shard_count = 40,
    deps = [
        ":test_macros_header",
        ":test_utils",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "half_test",
    srcs = ["half_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        ":test_macros_header",
        ":test_utils",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "slice_test",
    srcs = ["slice_test.cc"],
    shard_count = 40,
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "multidimensional_slice_test",
    srcs = ["multidimensional_slice_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "dynamic_ops_test",
    timeout = "moderate",
    srcs = ["dynamic_ops_test.cc"],
    deps = [
        ":client_library_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
        "//tensorflow/stream_executor:device_memory_allocator",
    ],
)

xla_test(
    name = "tuple_test",
    srcs = ["tuple_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

xla_test(
    name = "vector_ops_reduce_test",
    srcs = ["vector_ops_reduce_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "reduce_test",
    srcs = ["reduce_test.cc"],
    shard_count = 31,
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

# External xla_test targets can add "reduce_window_test_library" to xla_test_library_deps, in order
# to refer to the cc_library compiled with the correct backend macros. The following test target
# "reduce_window_test" is an example.
xla_test_library(
    name = "reduce_window_test_library",
    srcs = ["reduce_window_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "reduce_window_test",
    timeout = "long",
    srcs = [],
    shard_count = 20,
    tags = [
        "optonly",
    ],
    xla_test_library_deps = [":reduce_window_test_library"],
    deps = [":test_macros_header"],
)

xla_test(
    name = "select_and_scatter_test",
    timeout = "long",
    srcs = ["select_and_scatter_test.cc"],
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "copy_test",
    srcs = ["copy_test.cc"],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

xla_test(
    name = "reduce_hlo_test",
    srcs = ["reduce_hlo_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "token_hlo_test",
    srcs = ["token_hlo_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla/service:hlo_verifier",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "call_test",
    srcs = ["call_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "custom_call_test",
    srcs = ["custom_call_test.cc"],
    backends = ["cpu"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

xla_test(
    name = "binop_scaling_test",
    srcs = ["binop_scaling_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "broadcast_simple_test",
    srcs = ["broadcast_simple_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

xla_test(
    name = "pad_test",
    srcs = ["pad_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

xla_test(
    name = "fmax_fmin_test",
    srcs = ["fmax_fmin_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "log_test",
    srcs = ["log_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "matrix_ops_simple_test",
    srcs = ["matrix_ops_simple_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_test(
    name = "prng_test",
    timeout = "long",
    srcs = ["prng_test.cc"],
    shard_count = 6,
    # TODO(b/148276347) The test fails on macOS.
    tags = ["nomac"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "reshape_test",
    srcs = ["reshape_test.cc"],
    shard_count = 30,
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "reverse_test",
    srcs = ["reverse_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_test(
    name = "vector_ops_simple_test",
    srcs = ["vector_ops_simple_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "concat_test",
    srcs = ["concat_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array3d",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "convert_test",
    srcs = ["convert_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
    ],
)

xla_test(
    name = "all_reduce_test",
    srcs = ["all_reduce_test.cc"],
    blacklisted_backends = [
        # All reduce is not supported on the interpreter backend.
        "interpreter",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "collective_ops_test",
    srcs = ["collective_ops_test.cc"],
    args = ["--xla_force_host_platform_device_count=4"],
    backends = [
        "gpu",
        "cpu",
    ],
    tags = [
        # This test is tagged "manual" because it requires multiple GPUs, and
        # Forge only supports single-GPU tests.  Guitar skips "manual" tests
        # unless they're also tagged "guitar".
        "guitar",
        "manual",
        "multi_gpu",
        "no_oss",
        "notap",
    ],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/service/gpu:nccl_all_reduce_thunk",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "bitcast_convert_test",
    srcs = ["bitcast_convert_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "compilation_cache_test",
    srcs = ["compilation_cache_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "floor_ceil_test",
    srcs = ["floor_ceil_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "compute_constant_test",
    srcs = ["compute_constant_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "execution_profile_test",
    srcs = ["execution_profile_test.cc"],
    blacklisted_backends = [
        # Execution profiles are not supported on the interpreter backend.
        "interpreter",
    ],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "execution_profile_test_with_xla_hlo_profile",
    srcs = ["execution_profile_test.cc"],
    args = ["--xla_hlo_profile"],
    blacklisted_backends = [
        # Hlo profiles are not supported on the interpreter backend.
        "interpreter",
    ],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "replay_test",
    srcs = ["replay_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "broadcast_test",
    srcs = ["broadcast_test.cc"],
    deps = [
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "llvm_compiler_test",
    srcs = ["llvm_compiler_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":verified_hlo_module",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:llvm_compiler",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/gpu:nvptx_compiler_impl",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:core",
    ],
)

xla_test(
    name = "round_trip_packed_literal_test",
    srcs = ["round_trip_packed_literal_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:packed_literal_reader",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "cpu_gpu_fusion_test",
    srcs = ["cpu_gpu_fusion_test.cc"],
    deps = [
        ":client_library_test_base",
        ":hlo_test_base",
        ":literal_test_util",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//third_party/eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "multioutput_fusion_test",
    srcs = ["multioutput_fusion_test.cc"],
    backends = ["gpu"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "local_client_aot_test",
    srcs = [
        "local_client_aot_test.cc",
        ":local_client_aot_test_computation.o",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

xla_test(
    name = "local_client_allocation_test",
    srcs = ["local_client_allocation_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:local_client_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/types:optional",
    ],
)

xla_test(
    name = "local_client_execute_test",
    # TODO(b/79375911): Test times out in LLVM at normal size.
    size = "large",
    srcs = ["local_client_execute_test.cc"],
    shard_count = 30,
    tags = ["optonly"],
    deps = [
        ":literal_test_util",
        ":local_client_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
        "//tensorflow/stream_executor:device_memory_allocator",
    ],
)

xla_test(
    name = "outfeed_in_nested_computation_test",
    srcs = ["outfeed_in_nested_computation_test.cc"],
    blacklisted_backends = [
        # Outfeed ops are not supported on the interpreter backend.
        "interpreter",
    ],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla/tests:local_client_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "hlo_metadata_test",
    srcs = [
        "hlo_metadata_test.cc",
    ],
    deps = [
        ":local_client_test_base",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/core:test_main",
    ],
)

xla_test(
    name = "round_trip_transfer_test",
    srcs = ["round_trip_transfer_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "reshape_motion_test",
    srcs = ["reshape_motion_test.cc"],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:reference_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/types:span",
    ],
)

xla_test(
    name = "deep_graph_test",
    srcs = ["deep_graph_test.cc"],
    deps = [
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
    ],
)

tf_cc_test(
    name = "literal_test_util_test",
    srcs = ["literal_test_util_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

xla_test(
    name = "transfer_manager_test",
    srcs = ["transfer_manager_test.cc"],
    deps = [
        ":literal_test_util",
        ":local_client_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:stream_pool",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core:test",
        "//tensorflow/stream_executor:device_memory_allocator",
    ],
)

# A demo of textual IR based test.
xla_test(
    name = "sample_text_test",
    srcs = ["sample_text_test.cc"],
    # You can leave this empty if you want to test all supported backends.
    backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:optional",
    ],
)

# A demo of test that loads an hlo module from a file and compares results on gpu and cpu.
tf_cc_test(
    name = "sample_file_test",
    srcs = ["sample_file_test.cc"],
    data = ["isolated_convolution.hlo"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":hlo_test_base",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/service:cpu_plugin",  # reference backend
        "//tensorflow/compiler/xla/service:gpu_plugin",  # test backend
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.cc"],
    # There is nothing backend specific in this test, so just pick an arbitrary backend.
    backends = ["cpu"],
    deps = [
        ":local_client_test_base",
        ":test_macros_header",
        ":test_utils",
        ":xla_internal_test_main",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_test(
    name = "iota_test",
    srcs = ["iota_test.cc"],
    shard_count = 30,
    tags = [
        # Require optimized builds, iota_test_cpu is very slow in fastbuild.
        "optonly",
    ],
    deps = [
        ":client_library_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "multiple_devices_on_host_test",
    srcs = ["multiple_devices_on_host_test.cc"],
    args = ["--xla_force_host_platform_device_count=4"],
    deps = [
        ":xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_test(
    name = "ptxas_bug_120501638",
    srcs = ["ptxas_bug_120501638.cc"],
    tags = [
        # Disabled in OSS until nvidia publicly releases a fixed ptxas.
        "no_oss",
    ],
    deps = [
        ":hlo_test_base",
        ":test_macros_header",
        ":xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:test",
    ],
)

xla_test(
    name = "triangular_solve_test",
    srcs = ["triangular_solve_test.cc"],
    tags = [
        "enable_for_xla_interpreter",
        "noasan",  # sometimes times out, http://b/78650012
    ],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)

xla_test(
    name = "cholesky_test",
    srcs = ["cholesky_test.cc"],
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":test_macros_header",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/core:test",
    ],
)
