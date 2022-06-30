load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//smartass/brain/inference/...",
        "//tensorflow/core/tfrt/saved_model/...",
        "//tensorflow/core/tfrt/tfrt_session/...",
    ],
)

cc_library(
    name = "graph_execution_options",
    srcs = ["graph_execution_options.cc"],
    hdrs = ["graph_execution_options.h"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/runtime",
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
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt_request_context",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_op_handler",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/tpu:tpu_resources",
        "//tensorflow/core/tfrt/utils",
        "//tensorflow/core/tfrt/utils:error_util",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "//tensorflow/core/tfrt/utils:tfrt_graph_execution_state",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "graph_executor_test",
    srcs = ["graph_executor_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":graph_executor",
        "//tensorflow/cc:array_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:while_loop",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/grappler/utils:grappler_test",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
