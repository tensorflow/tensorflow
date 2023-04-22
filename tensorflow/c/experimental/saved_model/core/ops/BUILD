load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# This package contains written convenience helpers for Eager Operations
# used by SavedModel. Once we autogenerate C++ Eager Op wrappers, we can remove these.
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = [
        # Restricting visibility for now
        "//tensorflow/c/experimental/saved_model/core:__subpackages__",
        "//tensorflow/c/experimental/saved_model/internal:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "restore_ops",
    srcs = [
        "restore_ops.cc",
    ],
    hdrs = [
        "restore_ops.h",
    ],
    deps = [
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
    ],
)

cc_library(
    name = "variable_ops",
    srcs = [
        "variable_ops.cc",
    ],
    hdrs = [
        "variable_ops.h",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "restore_ops_test",
    srcs = [
        "restore_ops_test.cc",
    ],
    data = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":restore_ops",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core:test_utils",
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
    ],
)

tf_cc_test(
    name = "variable_ops_test",
    srcs = [
        "variable_ops_test.cc",
    ],
    deps = [
        ":variable_ops",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core:test_utils",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
    ],
)
