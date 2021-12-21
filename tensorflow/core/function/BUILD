load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = ["//tensorflow/core/function:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "runtime_client",
    srcs = [
        "runtime_client.cc",
    ],
    hdrs = [
        "runtime_client.h",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:portable_jpeg_internal",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:op_def_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "runtime_client_test",
    srcs = ["runtime_client_test.cc"],
    deps = [
        ":runtime_client",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
