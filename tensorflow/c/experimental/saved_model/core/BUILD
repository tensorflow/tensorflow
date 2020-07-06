# Experimental SavedModel C APIs for TensorFlow. See RFC
# https://github.com/tensorflow/community/pull/207
# Targets in this directory are pure C++ "Classes" underlying the C API types
# under tf/c/experimental/saved_model/public/. They are subject to change and
# have visibility limited to Tensorflow's implementation only.
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = [
        "//tensorflow/c:__subpackages__",
        "//tensorflow/c/experimental/saved_model/internal:__pkg__",
        "//tensorflow/core:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "concrete_function",
    srcs = [
        "concrete_function.cc",
    ],
    hdrs = [
        "concrete_function.h",
    ],
    deps = [
        ":function_metadata",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "function_metadata",
    hdrs = [
        "function_metadata.h",
    ],
)

cc_library(
    name = "saved_model_api",
    hdrs = [
        "saved_model_api.h",
    ],
    deps = [
        ":concrete_function",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "saved_model_utils",
    srcs = [
        "saved_model_utils.cc",
    ],
    hdrs = [
        "saved_model_utils.h",
    ],
    deps = [
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/experimental/saved_model/core/revived_types:constant",
        "//tensorflow/c/experimental/saved_model/core/revived_types:variable",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "test_utils",
    testonly = True,
    srcs = [
        "test_utils.cc",
    ],
    hdrs = [
        "test_utils.h",
    ],
    deps = [
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime/eager:context",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tf_saved_model_impl",
    srcs = [
        "tf_saved_model_impl.cc",
    ],
    hdrs = ["tf_saved_model_impl.h"],
    deps = [
        ":concrete_function",
        ":saved_model_api",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/eager:context",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "pywrap_required_hdrs",
    textual_hdrs = [
        "concrete_function.h",
        "function_metadata.h",
        "saved_model_api.h",
    ],
    visibility = ["//tensorflow/python:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "concrete_function.cc",
        "concrete_function.h",
        "function_metadata.h",
        "saved_model_api.h",
        "tf_saved_model_impl.cc",
        "tf_saved_model_impl.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

tf_cc_test(
    name = "constant_loading_test",
    srcs = [
        "constant_loading_test.cc",
    ],
    deps = [
        ":saved_model_utils",
        ":test_utils",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core/revived_types:constant",
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
    name = "saved_variable_loading_test",
    srcs = [
        "saved_variable_loading_test.cc",
    ],
    deps = [
        ":saved_model_utils",
        ":test_utils",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/saved_model/core/revived_types:constant",
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
