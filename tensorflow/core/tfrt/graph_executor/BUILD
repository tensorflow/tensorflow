load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//smartass/brain/inference/...",
        "//tensorflow/core/tfrt/...",
    ],
)

cc_library(
    name = "graph_execution_options",
    srcs = ["graph_execution_options.cc"],
    hdrs = ["graph_execution_options.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":config",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/utils:bridge_graph_analysis",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "graph_executor",
    srcs = ["graph_executor.cc"],
    hdrs = ["graph_executor.h"],
    tags = ["no_oss"],
    deps = [
        ":graph_execution_options",
        "//learning/brain/experimental/tfrt/mlrt/application/tensorflow/compiler/transforms:import_model",
        "//learning/brain/experimental/tfrt/mlrt/application/tensorflow/kernel:context",
        "//learning/brain/experimental/tfrt/native_lowering/kernels:sync_context",
        "//learning/brain/experimental/tfrt/native_lowering/saved_model:saved_model_translate",
        "//learning/infra/mira/mlrt/bytecode",
        "//learning/infra/mira/mlrt/bytecode:executable",
        "//learning/infra/mira/mlrt/interpreter:context",
        "//learning/infra/mira/mlrt/interpreter:execute",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt_request_context",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/compiler/mlir/tfrt:transforms/update_op_cost_in_tfrt_mlir",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/utils",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "//tensorflow/core/tfrt/utils:tfrt_graph_execution_state",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "graph_executor_test",
    srcs = ["graph_executor_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":graph_executor",
        "//learning/brain/experimental/tfrt/mlrt/application/tensorflow/kernel",
        "//learning/brain/experimental/tfrt/native_lowering/kernels:kernels_alwayslink",
        "//learning/infra/mira/mlrt/interpreter:context",
        "//learning/infra/mira/mlrt/interpreter:value",
        "//tensorflow/cc:array_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:const_op",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/grappler/utils:grappler_test",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:tensor",
        "@tf_runtime//cpp_tests:common",
    ],
)

cc_library(
    name = "synchronous_graph_executor",
    srcs = ["synchronous_graph_executor.cc"],
    hdrs = ["synchronous_graph_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_executor",
        "//learning/brain/experimental/tfrt/mlrt/application/tensorflow/kernel",
        "//learning/brain/experimental/tfrt/native_lowering/kernels",
        "//learning/brain/experimental/tfrt/native_lowering/kernels:kernels_alwayslink",
        "//learning/infra/mira/mlrt/interpreter:context",
        "//learning/infra/mira/mlrt/interpreter:value",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/utils:error_util",
        "@com_google_absl//absl/status:statusor",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":config_proto_cc",
        "//google/protobuf:any_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

tf_cc_test(
    name = "synchronous_graph_executor_test",
    srcs = ["synchronous_graph_executor_test.cc"],
    deps = [
        ":synchronous_graph_executor",
        "//learning/brain/experimental/tfrt/mlrt/application/tensorflow/kernel",
        "//learning/brain/experimental/tfrt/native_lowering/kernels:kernels_alwayslink",
        "//learning/infra/mira/mlrt/interpreter:value",
        "//tensorflow/cc:array_ops",
        "//tensorflow/core/tfrt/utils:error_util",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//cpp_tests:common",
    ],
)

tf_proto_library(
    name = "config_proto",
    srcs = ["config.proto"],
    protodeps = ["//google/protobuf:any"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "test_config_proto",
    testonly = True,
    srcs = ["test_config.proto"],
)

tf_cc_test(
    name = "config_test",
    srcs = ["config_test.cc"],
    deps = [
        ":config",
        ":config_proto_cc",
        ":test_config_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)
