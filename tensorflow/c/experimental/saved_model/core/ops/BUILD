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
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "owned_eager_op",
    hdrs = [
        "owned_eager_op.h",
    ],
    deps = [
        "//tensorflow/c/eager:operation_interface",
    ],
)

cc_library(
    name = "owned_tensor_handle",
    hdrs = [
        "owned_tensor_handle.h",
    ],
    deps = [
        "//tensorflow/c/eager:tensor_handle_interface",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
    ],
)

cc_library(
    name = "owned_eager_context",
    hdrs = ["owned_eager_context.h"],
    deps = [
        "//tensorflow/c/eager:context_interface",
        "//tensorflow/core/common_runtime/eager:context",
    ],
)

cc_library(
    name = "owned_tensor",
    hdrs = ["owned_tensor.h"],
    deps = [
        "//tensorflow/c:tensor_interface",
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
        ":owned_eager_op",
        ":owned_tensor_handle",
        "//tensorflow/c/eager:context_interface",
        "//tensorflow/c/eager:tensor_handle_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "variable_ops_test",
    srcs = [
        "variable_ops_test.cc",
    ],
    deps = [
        ":owned_eager_context",
        ":owned_tensor",
        ":owned_tensor_handle",
        ":variable_ops",
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
