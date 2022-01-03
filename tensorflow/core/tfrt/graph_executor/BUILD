package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/core/tfrt/saved_model/...",
    ],
)

cc_library(
    name = "graph_execution_options",
    hdrs = ["graph_execution_options.h"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tfrt/runtime",
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
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tensorflow:upgrade_graph",
        "//tensorflow/compiler/mlir/tfrt:tf_cpurt_request_context",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
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
        "//tensorflow/core/runtime_fallback/opdefs:tfrt_fallback_util",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/runtime_fallback/util:tensor_util",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/utils",
        "//tensorflow/core/tfrt/utils:error_util",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "//tensorflow/core/tfrt/utils:tfrt_graph_execution_state",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)
