load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_cuda_cc_test", "tf_openmp_copts")
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_tensor_coding_deps",
    "tf_proto_library",
    "tf_proto_library_cc",
)
load("//tensorflow/compiler/xla:xla.bzl", "xla_py_proto_library")
load("//tensorflow:tensorflow.bzl", "tf_portable_proto_library")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//tensorflow/compiler/xla/service/cpu:build_defs.bzl", "runtime_copts")

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/compiler/aot/...",
        "//tensorflow/compiler/jit/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tests/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/python/compiler/...",
    ],
)

package_group(
    name = "friends",
    includes = [":internal"],
    packages = [
        "//learning/brain/tools/tf_replay/...",
        "//tensorflow/...",
        "//tensorflow_models/...",
        "//third_party/mlperf/submissions/training/v0_7/models/...",
    ],
)

cc_library(
    name = "tf2xla_supported_ops_lib",
    srcs = ["tf2xla_supported_ops.cc"],
    hdrs = ["tf2xla_supported_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_binary(
    name = "tf2xla_supported_ops",
    srcs = ["tf2xla_supported_ops_main.cc"],
    visibility = ["//visibility:public"],
    deps = [":tf2xla_supported_ops_lib"],
)

tf_proto_library(
    name = "tf2xla_proto",
    srcs = ["tf2xla.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

# A proto library that is minimal in size and dependencies for platforms like Android.
tf_portable_proto_library(
    name = "portable_tf2xla_proto",
    config_string = "allow_all:true",
    header_outs = ["//tensorflow/compiler/tf2xla/tf2xla.proto.h"],
    portable_deps = ["//tensorflow/core:portable_proto_lib_full_runtime"],
    proto_deps = [
        ":tf2xla_proto",
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

xla_py_proto_library(
    name = "tf2xla_py",
    has_services = False,
    api_version = 2,
    visibility = ["//visibility:public"],
    deps = [":tf2xla_proto"],
)

tf_proto_library_cc(
    name = "host_compute_metadata_proto",
    srcs = ["host_compute_metadata.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "graph_compiler_util",
    srcs = ["graph_compiler_util.cc"],
    hdrs = ["graph_compiler_util.h"],
    visibility = [":friends"],
    deps = [
        ":functionalize_control_flow",
        ":sharding_util",
        ":tf2xla_proto_cc",
        ":tf2xla_util",
        ":xla_compiler",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tf2xla",
    srcs = ["tf2xla.cc"],
    hdrs = ["tf2xla.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":graph_compiler_util",
        ":tf2xla_proto_cc",
        ":tf2xla_util",
        ":xla_compiler",
        "//tensorflow/compiler/aot:aot_only_var_handle_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mlir_tf2xla",
    srcs = ["mlir_tf2xla.cc"],
    hdrs = ["tf2xla.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf2xla_proto_cc",
        ":tf2xla_util",
        ":xla_compiler",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:compile_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/xla:hlo",
        "//tensorflow/compiler/mlir/xla:mlir_hlo_to_hlo",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)

# The filegroups below are explicitly used by
# tensorflow/tools/pip_package:build_pip_package to ensure we include the proper
# sources for the XLA AOT CPU runtime; as these are necessary outside of bazel
# when linking tfcompile objects using saved_model_cli (e.g. using the
# tensorflow pip package). The associated .cc files are included in tensorflow
# pip package's xla_aot_runtime_srcs/ subdirectory. All necessary headers are
# also included in the pip package's include/tensorflow/ and include/external/
# subdirectories. Note however that sometimes additional object files may need
# to be linked when linking aot xla objects, e.g. abseil libraries. See the deps
# attribute of the "xla_compiled_cpu_runtime_standalone" target below for an
# exhaustive list.
filegroup(
    name = "xla_compiled_cpu_runtime_hdrs",
    srcs = [
        "xla_compiled_cpu_function.h",
        "//tensorflow/compiler/xla:cpu_runtime_hdrs",
        "//tensorflow/compiler/xla/service/cpu:single_threaded_runtime_hdrs",
        "//tensorflow/core/kernels:xla_cpu_runtime_hdrs",
        "//tensorflow/core/platform:xla_cpu_runtime_srcs",
    ],
    visibility = ["//tensorflow/tools/pip_package:__pkg__"],
)

filegroup(
    name = "xla_compiled_cpu_runtime_srcs",
    srcs = [
        "xla_compiled_cpu_function.cc",
        "//tensorflow/compiler/xla:cpu_runtime_srcs",
        "//tensorflow/compiler/xla/service/cpu:single_threaded_runtime_srcs",
        "//tensorflow/core/kernels:xla_cpu_runtime_srcs",
        "//tensorflow/core/platform:xla_cpu_runtime_srcs",
    ],
    visibility = ["//tensorflow/tools/pip_package:__pkg__"],
)

# This stand-alone target is used to ensure that we can build tf_library type
# targets against the subset of sources declared in
# xla_compiled_cpu_runtime_{srcs,hdrs}.
#
# The macros in tensorflow/python/tools/tools.bzl produce AOT compiled binaries
# that rely on this target, as do unit tests in tensorflow/python/tools.
#
# See above for the significance of the source filegroups.
cc_library(
    name = "xla_compiled_cpu_runtime_standalone",
    srcs = [
        ":xla_compiled_cpu_runtime_srcs",
    ],
    hdrs = [
        ":xla_compiled_cpu_runtime_hdrs",
    ],
    copts = runtime_copts() + tf_openmp_copts(),
    features = ["fully_static_link"],
    linkstatic = 1,
    visibility = [":friends"],
    # Note, we specifically removed MKL and multithreaded dependencies so the
    # standalone does not require the MKL binary blob or threading libraries.
    #
    # TODO(ebrevdo): Remove tf_additoinal_tensor_coding_deps in favor of
    # absl/strings:cord when we update absl to a newer version.
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//third_party/eigen3",
        "//tensorflow/core/framework:numeric_types",
    ] + tf_additional_tensor_coding_deps(),
    alwayslink = 1,
)

cc_library(
    name = "xla_compiled_cpu_function",
    srcs = ["xla_compiled_cpu_function.cc"],
    hdrs = ["xla_compiled_cpu_function.h"],
    visibility = ["//visibility:public"],
    deps = [
        # Keep dependencies to a minimum here; this library is used in every AOT
        # binary produced by tfcompile.
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/core/platform:types",
    ],
)

tf_cc_test(
    name = "cpu_function_runtime_test",
    srcs = ["cpu_function_runtime_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "xla_jit_compiled_cpu_function",
    srcs = ["xla_jit_compiled_cpu_function.cc"],
    hdrs = ["xla_jit_compiled_cpu_function.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf2xla",
        ":tf2xla_proto_cc",
        ":xla_compiled_cpu_function",
        "//tensorflow/compiler/xla:cpu_function_runtime",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service/cpu:buffer_info_util",
        "//tensorflow/compiler/xla/service/cpu:cpu_executable",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor:platform",
    ],
)

cc_library(
    name = "xla_compiler",
    srcs = [
        "const_analysis.cc",
        "graph_compiler.cc",
        "xla_compilation_device.cc",
        "xla_compiler.cc",
        "xla_context.cc",
        "xla_expression.cc",
        "xla_helpers.cc",
        "xla_op_kernel.cc",
        "xla_op_registry.cc",
        "xla_resource.cc",
        "xla_cpu_backend.cc",
    ] + if_cuda_is_configured([
        "xla_gpu_backend.cc",
    ]) + if_rocm_is_configured([
        "xla_gpu_backend.cc",
    ]),
    hdrs = [
        "const_analysis.h",
        "graph_compiler.h",
        "xla_compilation_device.h",
        "xla_compiler.h",
        "xla_context.h",
        "xla_expression.h",
        "xla_helpers.h",
        "xla_op_kernel.h",
        "xla_op_registry.h",
        "xla_resource.h",
    ],
    visibility = [":friends"],
    deps = [
        ":common",
        ":frontend_attributes_util",
        ":host_compute_metadata_proto_cc",
        ":rearrange_function_argument",
        ":sharding_util",
        ":side_effect_util",
        ":tf2xla_util",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/jit:xla_cluster_util",
        "//tensorflow/compiler/tf2xla/lib:util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/client/lib:constants",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:stream_executor_no_cuda",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
    alwayslink = 1,
)

cc_library(
    name = "common",
    srcs = [
        "literal_util.cc",
        "shape_util.cc",
        "type_util.cc",
    ],
    hdrs = [
        "literal_util.h",
        "shape_util.h",
        "type_util.h",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "frontend_attributes_util",
    srcs = ["frontend_attributes_util.cc"],
    hdrs = ["frontend_attributes_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "sharding_util",
    srcs = ["sharding_util.cc"],
    hdrs = ["sharding_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "sharding_util_test",
    srcs = ["sharding_util_test.cc"],
    deps = [
        ":sharding_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Internal targets below this point.

cc_library(
    name = "tf2xla_util",
    srcs = ["tf2xla_util.cc"],
    hdrs = ["tf2xla_util.h"],
    visibility = [":friends"],
    deps = [
        ":sharding_util",
        ":tf2xla_proto_cc",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "tf2xla_util_test",
    srcs = ["tf2xla_util_test.cc"],
    deps = [
        ":sharding_util",
        ":tf2xla_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tf2xla_test",
    srcs = ["tf2xla_test.cc"],
    deps = [
        ":tf2xla",
        ":tf2xla_proto_cc",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "xla_jit_compiled_cpu_function_test",
    srcs = ["xla_jit_compiled_cpu_function_test.cc"],
    deps = [
        ":tf2xla_proto_cc",
        ":xla_jit_compiled_cpu_function",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor:multi_platform_manager",
        "//tensorflow/stream_executor:platform",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "xla_compiler_test",
    srcs = [
        "xla_compiler_test.cc",
        "xla_expression_test.cc",
    ],
    deps = [
        ":common",
        ":side_effect_util",
        ":xla_compiler",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "literal_util_test",
    srcs = [
        "literal_util_test.cc",
    ],
    deps = [
        ":common",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "const_analysis_test",
    size = "small",
    srcs = ["const_analysis_test.cc"],
    deps = [
        ":xla_compiler",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/jit:xla_cluster_util",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "functionalize_control_flow_util",
    srcs = [
        "functionalize_control_flow_util.cc",
    ],
    hdrs = [
        "functionalize_control_flow_util.h",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "functionalize_cond",
    srcs = [
        "functionalize_cond.cc",
    ],
    hdrs = [
        "functionalize_cond.h",
    ],
    deps = [
        ":functionalize_control_flow_util",
        ":tf2xla_util",
        "//tensorflow/compiler/jit:union_find",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "functionalize_control_flow",
    srcs = [
        "functionalize_control_flow.cc",
    ],
    hdrs = [
        "functionalize_control_flow.h",
    ],
    deps = [
        ":functionalize_cond",
        ":functionalize_control_flow_util",
        ":functionalize_while",
        ":tf2xla_util",
        "//tensorflow/compiler/jit:union_find",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "mlir_bridge_pass",
    srcs = ["mlir_bridge_pass.cc"],
    hdrs = ["mlir_bridge_pass.h"],
    deps = [
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:core_cpu",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_bridge_pass_registration",
    srcs = [
        "mlir_bridge_pass_registration.cc",
    ],
    deps = [
        ":mlir_bridge_pass",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass_registration",
        "//tensorflow/core:core_cpu",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rearrange_function_argument",
    srcs = [
        "rearrange_function_argument.cc",
    ],
    hdrs = [
        "rearrange_function_argument.h",
    ],
    deps = [
        ":tf2xla_util",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "functionalize_control_flow_pass_registration",
    srcs = [
        "functionalize_control_flow_pass_registration.cc",
    ],
    visibility = [":friends"],
    deps = [
        ":functionalize_control_flow",
    ],
    alwayslink = 1,
)

cc_library(
    name = "functionalize_while",
    srcs = [
        "functionalize_while.cc",
    ],
    hdrs = [
        "functionalize_while.h",
    ],
    deps = [
        ":frontend_attributes_util",
        ":functionalize_cond",
        ":functionalize_control_flow_util",
        ":tf2xla_util",
        "//tensorflow/compiler/jit:union_find",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "functionalize_control_flow_test",
    srcs = ["functionalize_control_flow_test.cc"],
    deps = [
        ":functionalize_control_flow",
        ":test_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/compiler/tf2xla/cc:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:resource_variable_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "functionalize_cond_test",
    srcs = ["functionalize_cond_test.cc"],
    deps = [
        ":functionalize_cond",
        ":functionalize_control_flow",
        ":test_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/tf2xla/cc:xla_ops",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:resource_variable_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:test",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "xla_op_registry_test",
    srcs = ["xla_op_registry_test.cc"],
    deps = [
        ":xla_compiler",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "resource_operation_table",
    srcs = ["resource_operation_table.cc"],
    hdrs = ["resource_operation_table.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "resource_operation_table_test",
    srcs = ["resource_operation_table_test.cc"],
    deps = [
        ":resource_operation_table",
        ":xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "side_effect_util",
    srcs = ["side_effect_util.cc"],
    hdrs = ["side_effect_util.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:core_cpu",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "fused_batchnorm_reserve_space_test",
    size = "medium",
    srcs = ["fused_batchnorm_reserve_space_test.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/jit",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//third_party/eigen3",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "resource_util",
    srcs = ["resource_util.cc"],
    hdrs = ["resource_util.h"],
    visibility = [":friends"],
    deps = [
        ":resource_operation_table",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "resource_util_test",
    size = "small",
    srcs = ["resource_util_test.cc"],
    deps = [
        ":resource_util",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)
