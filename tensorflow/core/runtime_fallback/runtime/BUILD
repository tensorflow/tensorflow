load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/runtime_fallback/runtime/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

cc_library(
    name = "op_logger",
    hdrs = [
        "op_logger.h",
    ],
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/cpp_tests/saved_model:__subpackages__",
        "//tensorflow/core/runtime_fallback:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "runtime_fallback_alwayslink",
    srcs = [
        "conversion_function.cc",
        "runtime_fallback_batch_tf_opkernels.cc",
        "runtime_fallback_kernels.cc",
        "runtime_fallback_op_handler.cc",
        "static_registration.cc",
    ],
    hdrs = [
        "conversion_function.h",
        "runtime_fallback_kernels.h",
        "runtime_fallback_op_handler.h",
        "runtime_fallback_tensor.h",
    ],
    includes = [
        "third_party/tf_runtime/include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":fallback_batch_kernel",
        ":kernel_utils",
        ":op_logger",
        ":runtime_fallback_tensor",
        "//tensorflow/core/kernels/batching_util:adaptive_shared_batch_scheduler",
        "//tensorflow/core/kernels/batching_util:batch_scheduler_hdrs",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//tensorflow/core/runtime_fallback/util:attr_util",
        "//tensorflow/core/runtime_fallback/util:tensor_util",
        "//tensorflow/core/runtime_fallback/util:type_util",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/utils:error_util",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "//tensorflow/core/tfrt/utils:tensor_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/c:c_api_internal",
            "//tensorflow/c:tensor_interface",
            "//tensorflow/c:tf_datatype",
            "//tensorflow/c:tf_status",
            "//tensorflow/c:tf_tensor_internal",
            "//tensorflow/c/eager:abstract_operation",
            "//tensorflow/c/eager:abstract_tensor_handle",
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:session_options",
            "//tensorflow/core/common_runtime:core_cpu_impl",
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:core",  # Needed due to circular dep
            "//tensorflow/core/common_runtime/eager:eager_operation",
            "//tensorflow/core/common_runtime/eager:execute",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/framework:node_def_proto_cc",
            "//tensorflow/core/framework:tensor",
            "//tensorflow/core/kernels/batching_util:batch_resource_base",
            "//tensorflow/core/kernels/batching_util:bounded_executor",
            "//tensorflow/core/platform:errors",
            "//tensorflow/core/platform:random",
            "//tensorflow/core/platform:status",
            "//tensorflow/core/profiler/lib:traceme",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "runtime_fallback_tensor",
    srcs = ["runtime_fallback_tensor.cc"],
    hdrs = ["runtime_fallback_tensor.h"],
    visibility = [
        ":internal",
        "//tensorflow/core/runtime_fallback/conversion:__subpackages__",
    ],
    deps = [
        ":kernel_utils",
        "//tensorflow/core/runtime_fallback/util:tensor_util",
        "//tensorflow/core/runtime_fallback/util:type_util",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/c:tensor_interface",
            "//tensorflow/c:tf_datatype",
            "//tensorflow/c:tf_tensor",
            "//tensorflow/c:tf_tensor_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/framework:tensor",
        ],
    }),
)

cc_library(
    name = "kernel_utils",
    srcs = ["kernel_utils.cc"],
    hdrs = ["kernel_utils.h"],
    visibility = [
        ":internal",
        "//tensorflow/compiler/mlir/tfrt:__pkg__",
        "//tensorflow/core/runtime_fallback:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/c:tf_tensor",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:session_options",
            "//tensorflow/core/common_runtime:device_mgr",
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:context_distributed_manager",
            "//tensorflow/core/common_runtime/eager:core_no_xla",
            "//tensorflow/core/common_runtime/eager:eager_operation",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/platform:status",
        ],
    }),
)

cc_library(
    name = "fallback_batch_kernel",
    srcs = ["fallback_batch_kernel.cc"],
    hdrs = ["fallback_batch_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:op_requires",
        "//tensorflow/core/kernels:batch_kernels",
        "//tensorflow/core/kernels/batching_util:adaptive_shared_batch_scheduler",
        "//tensorflow/core/kernels/batching_util:batch_resource_base",
        "//tensorflow/core/kernels/batching_util:batch_scheduler_hdrs",
        "//tensorflow/core/kernels/batching_util:bounded_executor",
        "//tensorflow/core/kernels/batching_util:warmup",
        "//tensorflow/core/lib/core:refcount",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:random",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "//tensorflow/core/tfrt/utils:error_util",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@tf_runtime//:hostcontext",
    ],
)
