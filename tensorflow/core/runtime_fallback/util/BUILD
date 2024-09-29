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
        "//tensorflow/compiler/mlir/tfrt/tests/tfrt_fallback/...",
        "//tensorflow/core/runtime_fallback/...",
        "//tensorflow/core/tfrt/utils/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/utils:tensor_util",
        "//third_party/protobuf",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:errors",
        "@tf_runtime//:bef",
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
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:Support",
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
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//tensorflow/core/runtime_fallback/runtime:kernel_utils",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "@local_tsl//tsl/platform:status",
        "@tf_runtime//:hostcontext",
    ],
)

tf_cuda_library(
    name = "tensor_metadata",
    hdrs = ["tensor_metadata.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":type_util",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:tensor_shape",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
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
        ":tensor_metadata",
        ":type_util",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/core/framework:tensor_shape",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:tstring",
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
            "//tensorflow/c:tf_tensor_internal",
            "//tensorflow/core/framework:tensor",
            "//tensorflow/core/platform:status",
        ],
    }),
)

tf_cc_test(
    name = "type_util_test",
    srcs = ["type_util_test.cc"],
    deps = [
        ":type_util",
        "//tensorflow/core/framework:types_proto_cc",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
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
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/core/framework:tensor_shape",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:protobuf",
        "//third_party/protobuf",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
        "@eigen_archive//:eigen3_test",
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
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:bef",
        "@tf_runtime//:bef_attr_encoder",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
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
