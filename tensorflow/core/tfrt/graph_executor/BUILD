# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_shared_object", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/native_lowering/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/serving/servables/tfrt/...",
        # copybara:uncomment "//smartass/brain/inference/...",
        # copybara:uncomment "//tensorflow/compiler/mlir/tfrt/...",
        "//tensorflow/core/tfrt/...",
        "//tensorflow/core/tfrt/graph_executor/python/...",
        # copybara:uncomment "//tensorflow_serving/servables/tensorflow/...",
        # copybara:uncomment "//tensorflow_serving/servables/tensorflow/google/...",
    ],
)

cc_library(
    name = "graph_execution_options",
    srcs = ["graph_execution_options.cc"],
    hdrs = ["graph_execution_options.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/utils:bridge_graph_analysis",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
)

# For bazel python deps only
tf_cc_shared_object(
    name = "graph_execution_options.so",
    deps = [":graph_execution_options"],
)

cc_library(
    name = "executable_context",
    hdrs = ["executable_context.h"],
    deps = [
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "graph_executor",
    srcs = ["graph_executor.cc"],
    hdrs = ["graph_executor.h"],
    deps = [
        ":executable_context",
        ":export_mlir",
        ":graph_execution_options",
        ":sync_resource_state",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tfrt:backend_compiler",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/compiler/mlir/tfrt:transforms/update_op_cost_in_tfrt_mlir",
        "//tensorflow/compiler/mlir/tfrt/transforms/mlrt:import_model",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/public:version",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_utils",
        "//tensorflow/core/tfrt/common:metrics",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:executable",
        "//tensorflow/core/tfrt/mlrt/bytecode:function",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "//tensorflow/core/tfrt/mlrt/interpreter:execute",
        "//tensorflow/core/tfrt/mlrt/interpreter:value",
        "//tensorflow/core/tfrt/mlrt/kernel:context",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/runtime:step_id",
        "//tensorflow/core/tfrt/runtime:stream",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/stubs:tfrt_native_lowering_stub",
        "//tensorflow/core/tfrt/utils",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "//tensorflow/core/tfrt/utils:tfrt_graph_execution_state",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:refcount",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:support",
    ] + if_google(
        [
            "//learning/brain/experimental/tfrt/native_lowering/stubs:tfrt_native_lowering_impl",
        ],
    ),
)

tf_cc_test(
    name = "graph_executor_test",
    srcs = ["graph_executor_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":graph_execution_options",
        ":graph_executor",
        "//tensorflow/cc:array_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:common_shape_fns",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:op",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/grappler/utils:grappler_test",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "//tensorflow/core/tfrt/mlrt/interpreter:value",
        "//tensorflow/core/tfrt/mlrt/kernel",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
        "@tf_runtime//cpp_tests:common",
    ] + if_google(
        [
            "//learning/brain/experimental/tfrt/native_lowering/kernels:kernels_alwayslink",
        ],
    ),
)

cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":config_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

tf_proto_library(
    name = "config_proto",
    srcs = ["config.proto"],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "config_proto_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":config_proto"],
# )
# copybara:uncomment_end

tf_proto_library(
    name = "test_config_proto",
    testonly = True,
    srcs = ["test_config.proto"],
    visibility = if_google(
        [":friends"],
        ["//visibility:public"],
    ),
)

tf_cc_test(
    name = "config_test",
    srcs = ["config_test.cc"],
    deps = [
        ":config",
        ":config_proto_cc",
        ":test_config_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "sync_resource_state",
    hdrs = ["sync_resource_state.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/tfrt/utils:any_ptr",
        "@tf_runtime//:tensor",
    ],
)

cc_library(
    name = "export_mlir",
    hdrs = ["export_mlir.h"],
    deps = ["@llvm-project//mlir:IR"],
)
