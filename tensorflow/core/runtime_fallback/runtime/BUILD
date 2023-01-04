load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
)
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
    features = ["-layering_check"],
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
        "//tensorflow/core/tfrt/eager:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cuda_library(
    name = "runtime_fallback_alwayslink",
    srcs = [
        "conversion_function.cc",
        "runtime_fallback_batch_tf_opkernels.cc",
        "runtime_fallback_gpu_allocator.cc",
        "runtime_fallback_gpu_allocator.h",
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
    cuda_deps = [
        "//tensorflow/core/common_runtime/gpu:gpu_bfc_allocator",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime_impl",
        "//tensorflow/core/platform:mutex",
        "@tf_runtime//backends/gpu:gpu_config",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform",
        "@tf_runtime//backends/gpu:gpu_op_handler_alwayslink",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_gpu_alwayslink",
    ],
    includes = [
        "third_party/tf_runtime/include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":op_logger",
        ":kernel_utils",
        ":runtime_fallback_tensor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//tensorflow/core/runtime_fallback/util:attr_util",
        "//tensorflow/core/runtime_fallback/util:tensor_util",
        "//tensorflow/core/runtime_fallback/util:type_util",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/utils:error_util",
        "//tensorflow/core/tfrt/utils:tensor_util",
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
            "//tensorflow/core/common_runtime/eager:execute",
            "//tensorflow/core/common_runtime/eager:core",  # Needed due to circular dep
            "//tensorflow/core/common_runtime/eager:eager_operation",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/framework:node_def_proto_cc",
            "//tensorflow/core/framework:tensor",
            "//tensorflow/core/platform:errors",
            "//tensorflow/core/platform:random",
            "//tensorflow/core/platform:status",
            "//tensorflow/core/profiler/lib:traceme",
            "//tensorflow/core/kernels/batching_util:batch_resource_base",
            "//tensorflow/core/kernels/batching_util:bounded_executor",
        ],
    }),
    alwayslink = 1,
)

tf_cuda_library(
    name = "runtime_fallback_tensor",
    srcs = ["runtime_fallback_tensor.cc"],
    hdrs = ["runtime_fallback_tensor.h"],
    cuda_deps = [
        "@tf_runtime//backends/gpu:gpu_config",
        "@tf_runtime//backends/gpu:gpu_tensor",
    ],
    visibility = [
        ":internal",
        "//tensorflow/core/runtime_fallback/conversion:__subpackages__",
    ],
    deps = [
        ":kernel_utils",
        "@llvm-project//llvm:Support",
        "//tensorflow/core/runtime_fallback/util:tensor_util",
        "//tensorflow/core/runtime_fallback/util:type_util",
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
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__pkg__",
        "//tensorflow/core/runtime_fallback:__subpackages__",
        "//tensorflow/core/tfrt/eager:__subpackages__",
        "//tensorflow/core/tfrt/saved_model:__subpackages__",
    ],
    deps = [
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
            "//tensorflow/core/common_runtime:device_mgr",
            "//tensorflow/core/common_runtime/eager:context_distributed_manager",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:session_options",
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:eager_operation",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/platform:status",
        ],
    }),
)

tf_cuda_library(
    name = "runtime_fallback_gpu_alwayslink",
    srcs = [
        "gpu/conversion_function.cc",
        "gpu/static_registration.cc",
    ],
    hdrs = [
        "gpu/conversion_function.h",
    ],
    compatible_with = [],
    # Only build this library with --config=cuda.
    tags = [
        "manual",
        "no_oss",
        "requires_cuda",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kernel_utils",
        ":op_logger",
        ":runtime_fallback_tensor",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/runtime_fallback/util:attr_util",
        "//tensorflow/core/runtime_fallback/util:gpu_util",
        "//tensorflow/core/runtime_fallback/util:type_util",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/gpu:gpu_config",
        "@tf_runtime//backends/gpu:gpu_device",
        "@tf_runtime//backends/gpu:gpu_op_handler_alwayslink",
        "@tf_runtime//backends/gpu:gpu_tensor",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
        ],
    }),
    alwayslink = 1,
)
