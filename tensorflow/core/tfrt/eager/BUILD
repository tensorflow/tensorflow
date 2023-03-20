load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # copybara:uncomment "//learning/brain/experimental/dtensor/...",
        "//tensorflow/c/eager/...",
        "//tensorflow/core/runtime_fallback/...",  # TODO(chuanhao): remove after cl/326748977.
        "//tensorflow/core/tfrt/...",
        "//tensorflow/python/...",
    ],
)

cc_library(
    name = "transform_graph_function",
    srcs = [
        "transform_graph_function.cc",
    ],
    hdrs = [
        "transform_graph_function.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/common_runtime:function",
        "//tensorflow/core/common_runtime:function_body",
        "//tensorflow/core/common_runtime:function_optimization_registry",
        "//tensorflow/core/common_runtime:optimization_registry",
        "//tensorflow/core/common_runtime:optimize_function_graph_utils",
        "//tensorflow/core/common_runtime:placer",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "c_api_tfrt",
    srcs = [
        "c_api_tfrt.cc",
        "function_cache.cc",
        "op_cache.cc",
    ],
    hdrs = [
        "c_api_tfrt.h",
        "function_cache.h",
        "op_cache.h",
    ],
    deps = [
        ":core_runtime",
        ":tfrt_context",
        ":transform_graph_function",
        ":virtual_device",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:abstract_function",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:abstract_op_attrs",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core:saved_model_api",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:function",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/runtime_fallback/util:attr_util",
        "//tensorflow/core/runtime_fallback/util:tensor_util",
        "//tensorflow/core/tfrt/utils:error_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:attr_builder",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat_eager",
        "//tensorflow/core/runtime_fallback/runtime:op_logger",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/runtime_fallback/conversion:conversion_alwayslink",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "//tensorflow/core/tfrt/utils",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:dtype",
        "@tf_runtime//:bef_attr_encoder",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:metrics",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        "@tf_runtime//backends/common:eigentype",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/core/tfrt/eager/backends/cpu:cpu_registration_alwayslink",
    ] + if_cuda([
        "//tensorflow/core/tfrt/eager/backends/gpu:gpu_registration_alwayslink",
    ]),
)

cc_library(
    name = "tfrt_context",
    srcs = [
        "tfrt_context.cc",
    ],
    hdrs = [
        "tfrt_context.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":core_runtime",
        "//tensorflow/core/common_runtime:process_util",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/runtime_fallback/runtime:kernel_utils",
        "//tensorflow/core/tfrt/common:global_state",
        "//tensorflow/core/tpu:virtual_device",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/runtime_fallback/conversion:conversion_alwayslink",
        "//tensorflow/core/tfrt/runtime:tf_threadpool_concurrent_work_queue",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext_alwayslink",
        "//tensorflow/core/tfrt/eager/backends/cpu:cpu_registration_alwayslink",
    ] + if_cuda([
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_gpu_alwayslink",
        "//tensorflow/core/tfrt/eager/backends/gpu:gpu_registration_alwayslink",
    ]),
)

cc_library(
    name = "virtual_device",
    hdrs = ["virtual_device.h"],
    deps = [
        "@tf_runtime//:hostcontext",
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
    textual_hdrs = [
        "c_api_tfrt.h",
    ],
    visibility = [
        "//visibility:private",  # Only private by automation, not intent. Owner may accept CLs adding visibility. See go/scheuklappen#explicit-private.
    ],
)

alias(
    name = "core_runtime",
    actual = "//tensorflow/core/tfrt/eager/core_runtime:core_runtime_lib",
)

tf_cc_test(
    name = "function_cache_test",
    size = "small",
    srcs = [
        "function_cache_test.cc",
    ],
    args = ["--heap_check="],
    tags = ["no_oss"],
    deps = [
        ":c_api_tfrt",
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/experimental/ops:array_ops",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat_eager",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//backends/cpu:tf_ops_alwayslink",
    ],
)

tf_cc_test(
    name = "op_cache_test",
    size = "small",
    srcs = [
        "op_cache_test.cc",
    ],
    args = ["--heap_check="],
    tags = ["no_oss"],
    deps = [
        ":c_api_tfrt",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/types:span",
        "@tf_runtime//backends/cpu:core_runtime",
        "@tf_runtime//backends/cpu:tf_ops_alwayslink",
    ],
)
