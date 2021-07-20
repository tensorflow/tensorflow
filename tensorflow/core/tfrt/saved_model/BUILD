load("//tensorflow:tensorflow.bzl", "if_google")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/serving/...",
        "//tensorflow/core/runtime_fallback/...",
        "//tensorflow/core/tfrt/saved_model/tests/...",
        "//tensorflow_serving/...",
        # copybara:uncomment "//quality/webanswers/servo2/...",
    ],
)

# No native TFRT-based TF CPU op/kernels.
# This target is cheaper to build.
cc_library(
    name = "saved_model_fallback_only",
    srcs = [
        "saved_model.cc",
        "saved_model_import_input.cc",
        "saved_model_import_input.h",
    ],
    hdrs = ["saved_model.h"],
    tags = ["no_oss"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tensorflow:upgrade_graph",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:saved_model",
        "//tensorflow/compiler/mlir/tfrt:tf_cpurt_kernels_alwayslink",
        "//tensorflow/compiler/mlir/tfrt:tf_cpurt_request_context",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core/tfrt/runtime:tf_threadpool_concurrent_work_queue",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/utils:error_util",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_op_handler",
        "//tensorflow/core/runtime_fallback/runtime:kernel_utils",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/runtime_fallback/util:tensor_util",
        "//tensorflow/core/tfrt/tpu:tpu_resources",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/runtime:runtime_fallback_only",
        "//tensorflow/core/tfrt/utils",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "//tensorflow/core/tfrt/utils:model_metadata",
        "//tensorflow/core/tfrt/utils:statusor",
        "//tensorflow/core/tfrt/utils:tensor_util",
        "//tensorflow/core/tfrt/utils:tfrt_graph_execution_state",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:metrics",
        "@tf_runtime//:support",
    ] + if_google([
        "//third_party/tf_runtime_google:streamz_metrics_registry_alwayslink",
        "//third_party/tf_runtime_google:xprof_tracing_sink_alwayslink",
    ]),
)

cc_library(
    name = "saved_model",
    hdrs = ["saved_model.h"],
    tags = ["no_oss"],
    deps = [
        ":saved_model_fallback_only",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:saved_model",
        "//tensorflow/core:session_options",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/platform:threadpool_options",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_op_handler",
        "//tensorflow/core/runtime_fallback/runtime:kernel_utils",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/runtime:runtime_fallback_only",
        "//tensorflow/core/tfrt/utils:model_metadata",
        "//tensorflow/core/tfrt/utils:statusor",
        "//tensorflow/core/tfrt/utils:tfrt_graph_execution_state",
        # copybara:uncomment "//third_party/tf_runtime_google:streamz_metrics_registry_alwayslink",
        # copybara:uncomment "//third_party/tf_runtime_google:xprof_tracing_sink_alwayslink",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//backends/cpu:tf_ops_alwayslink",
    ],
)

cc_library(
    name = "saved_model_testutil_fallback_only",
    testonly = 1,
    srcs = ["saved_model_testutil.cc"],
    hdrs = ["saved_model_testutil.h"],
    tags = ["no_oss"],
    deps = [
        ":saved_model_fallback_only",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/platform:test_benchmark",
        "//tensorflow/core/tfrt/runtime:runtime_fallback_only",
        "@com_google_googletest//:gtest",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
    ],
)

cc_library(
    name = "saved_model_testutil",
    testonly = 1,
    hdrs = ["saved_model_testutil.h"],
    tags = ["no_oss"],
    deps = [
        ":saved_model_fallback_only",
        ":saved_model_testutil_fallback_only",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/core/platform:test_benchmark",
        "//tensorflow/core/tfrt/runtime",
        "@com_google_googletest//:gtest",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
    ],
)

cc_library(
    name = "saved_model_import_input",
    srcs = ["saved_model_import_input.cc"],
    hdrs = ["saved_model_import_input.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:upgrade_graph",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/utils:tfrt_graph_execution_state",
    ],
)
