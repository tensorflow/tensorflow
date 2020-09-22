load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# This package contains classes corresponding to Revived SavedObjectGraph types
# used by SavedModel. See https://cs.opensource.google/tensorflow/tensorflow/+/c575e2ba93c442121d98d3f125d83fed1339924d:tensorflow/core/protobuf/saved_object_graph.proto;l=56-62
package(
    default_visibility = [
        # Restricting visibility for now
        "//tensorflow/c/experimental/saved_model/core:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "asset",
    srcs = [
        "asset.cc",
    ],
    hdrs = [
        "asset.h",
    ],
    deps = [
        ":tensorhandle_convertible",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "constant",
    srcs = [
        "constant.cc",
    ],
    hdrs = [
        "constant.h",
    ],
    deps = [
        ":tensorhandle_convertible",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
    ],
)

cc_library(
    name = "flat_tensor_function",
    srcs = [
        "flat_tensor_function.cc",
    ],
    hdrs = [
        "flat_tensor_function.h",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:context",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "variable",
    srcs = [
        "variable.cc",
    ],
    hdrs = [
        "variable.h",
    ],
    deps = [
        ":tensorhandle_convertible",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core/ops:variable_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:context",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tensorhandle_convertible",
    hdrs = [
        "tensorhandle_convertible.h",
    ],
    deps = [
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
    ],
)

cc_library(
    name = "tf_concrete_function",
    srcs = [
        "tf_concrete_function.cc",
    ],
    hdrs = [
        "tf_concrete_function.h",
    ],
    deps = [
        ":flat_tensor_function",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core:concrete_function",
        "//tensorflow/c/experimental/saved_model/core:function_metadata",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:context",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tf_signature_def_function",
    srcs = [
        "tf_signature_def_function.cc",
    ],
    hdrs = [
        "tf_signature_def_function.h",
    ],
    deps = [
        ":flat_tensor_function",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core:signature_def_function",
        "//tensorflow/c/experimental/saved_model/core:signature_def_function_metadata",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:context",
        "@com_google_absl//absl/types:span",
    ],
)
