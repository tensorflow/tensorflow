load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_cc_test(
    name = "internal_test",
    srcs = ["internal_test.cc"],
    deps = [
        ":internal",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "internal",
    srcs = ["internal.cc"],
    hdrs = ["internal.h"],
    deps = [
        ":wrapper_operation",
        "//tensorflow/c:conversion_macros",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "wrapper_operation",
    srcs = ["wrapper_operation.cc"],
    hdrs = ["wrapper_operation.h"],
    deps = ["//tensorflow/c/eager:abstract_operation"],
)
