# Experimental SavedModel C APIs for TensorFlow. See RFC
# https://github.com/tensorflow/community/pull/207
# Targets in this directory are pure C++ "Classes" underlying the C API types
# under tf/c/experimental/saved_model/public/. They are subject to change and
# have visibility limited to Tensorflow's implementation only.

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
        "//tensorflow/c/eager:operation_interface",
        "//tensorflow/c/eager:tensor_handle_interface",
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
