load(
    "//tensorflow:tensorflow.bzl",
    "if_oss",
    "tf_cc_test",
    "tf_cuda_library",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package_group(
    name = "internal",
    packages = [
        "//learning/brain/experimental/tfrt/native_lowering/kernels/...",
        "//tensorflow/core/runtime_fallback/...",
        "//tensorflow/core/tfrt/eager/backends/tpu/...",
        "//tensorflow/core/tfrt/utils/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

cc_library(
    name = "attr_util",
    srcs = [
        "attr_util.cc",
    ],
    hdrs = [
        "attr_type.def",
        "attr_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "//tensorflow/core/tfrt/utils:tensor_util",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/framework:tensor",
            "//tensorflow/core/framework:tensor_proto_cc",
            "//tensorflow/core/framework:types_proto_cc",
        ],
    }),
)

cc_library(
    name = "type_util",
    hdrs = [
        "dtype.def",
        "type_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
        "//tensorflow/core/platform:logging",
        "@tf_runtime//:dtype",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core/framework:types_proto_cc",
        ],
    }),
)

cc_library(
    name = "fallback_test_util",
    srcs = ["fallback_test_util.cc"],
    hdrs = ["fallback_test_util.h"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt_request_context",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//tensorflow/core/runtime_fallback/runtime:kernel_utils",
        "@tf_runtime//:hostcontext",
    ],
)

tf_cuda_library(
    name = "tensor_util",
    srcs = ["tensor_util.cc"],
    hdrs = [
        "tensor_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":type_util",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/c:tf_tensor",
            "//tensorflow/c:tf_tensor_internal",
            "//tensorflow/core/framework:tensor",
            "//tensorflow/core/platform:status",
        ],
    }),
)

tf_cuda_library(
    name = "gpu_util",
    srcs = [
        "gpu/gpu_utils.cc",
    ],
    hdrs = [
        "gpu/gpu_utils.h",
    ],
    compatible_with = [],
    # Only build this library with --config=cuda.
    tags = [
        "manual",
        "no_oss",
        "requires_cuda",
    ],
    deps = [
        ":tensor_util",
        ":type_util",
        "@tf_runtime//:support",
        "@tf_runtime//backends/gpu:gpu_config",
        "@tf_runtime//backends/gpu:gpu_device",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/c:tf_tensor",
            "//tensorflow/c:tf_tensor_internal",
            "//tensorflow/compiler/xla/stream_executor:platform",
            "//tensorflow/compiler/xla/stream_executor/cuda:cuda_driver",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core/common_runtime/gpu:gpu_runtime",
        ],
    }),
)

tf_cc_test(
    name = "type_util_test",
    srcs = ["type_util_test.cc"],
    deps = [
        ":type_util",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:test",
            "//tensorflow/core:test_main",
            "//tensorflow/core/platform:status",
            "//tensorflow/core/platform:types",
        ],
    }),
)

tf_cc_test(
    name = "tensor_util_test",
    srcs = ["tensor_util_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":tensor_util",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:test",
            "//tensorflow/core:test_main",
            "//tensorflow/core/platform:status",
            "//tensorflow/core/platform:types",
        ],
    }),
)

tf_cc_test(
    name = "attr_util_test",
    srcs = ["attr_util_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":attr_util",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:test",
            "//tensorflow/core:test_main",
            "//tensorflow/core/platform:status",
            "//tensorflow/core/platform:types",
        ],
    }),
)
