# Experimental Implementation of SavedModel C APIs for TensorFlow. See RFC
# https://github.com/tensorflow/community/pull/207
# External clients should not worry about this directory; all contents are implementation details.
# Code in this directory is intended to form the glue between the C API and the internal C++
# implementation by
# 1. mapping C API calls onto correponding methods of C++ objects
# 2. mapping opaque C types onto C++ classes

# Note(bmzhao): The *.cc files in this directory form the direct implementation of the
# C API functions exposed in tf/c/experimental/saved_model/public/.

# Note(bmzhao): All *type.h files in this directory are the internal definitions of
# the opaque C types. These headers should only be visible to internal tensorflow
# implementors.
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
)

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "concrete_function",
    srcs = [
        "concrete_function.cc",
    ],
    hdrs = [
        "//tensorflow/c/experimental/saved_model/public:concrete_function.h",
    ],
    copts = tf_copts(),
    visibility = [
        "//tensorflow/c/experimental/saved_model/public:__pkg__",
    ],
    deps = [
        ":concrete_function_type",
        ":function_metadata",
        ":function_metadata_type",
        ":tensorhandle_list",
        ":tensorhandle_list_type",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:tfe_op_internal",
        "//tensorflow/c/experimental/saved_model/core:concrete_function",
        "//tensorflow/c/experimental/saved_model/core:function_metadata",
    ],
)

cc_library(
    name = "concrete_function_list",
    srcs = [
        "concrete_function_list.cc",
    ],
    hdrs = [
        "//tensorflow/c/experimental/saved_model/public:concrete_function_list.h",
    ],
    copts = tf_copts(),
    visibility = [
        "//tensorflow/c/experimental/saved_model/public:__pkg__",
    ],
    deps = [
        ":concrete_function",
        ":concrete_function_list_type",
        ":concrete_function_type",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c/experimental/saved_model/core:concrete_function",
    ],
)

cc_library(
    name = "concrete_function_list_type",
    hdrs = [
        "concrete_function_list_type.h",
    ],
    deps = [
        "//tensorflow/c/experimental/saved_model/core:concrete_function",
    ],
)

cc_library(
    name = "concrete_function_type",
    hdrs = [
        "concrete_function_type.h",
    ],
    deps = [
        "//tensorflow/c:conversion_macros",
        "//tensorflow/c/experimental/saved_model/core:concrete_function",
    ],
)

cc_library(
    name = "function_metadata",
    srcs = [
        "function_metadata.cc",
    ],
    hdrs = [
        "//tensorflow/c/experimental/saved_model/public:function_metadata.h",
    ],
    copts = tf_copts(),
    visibility = [
        "//tensorflow/c/experimental/saved_model/public:__pkg__",
    ],
    deps = [
        ":function_metadata_type",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c/experimental/saved_model/core:function_metadata",
    ],
)

cc_library(
    name = "function_metadata_type",
    hdrs = [
        "function_metadata_type.h",
    ],
    deps = [
        "//tensorflow/c:conversion_macros",
        "//tensorflow/c/experimental/saved_model/core:function_metadata",
    ],
)

cc_library(
    name = "saved_model_api",
    srcs = [
        "saved_model_api.cc",
    ],
    hdrs = [
        "//tensorflow/c/experimental/saved_model/public:saved_model_api.h",
    ],
    copts = tf_copts(),
    visibility = [
        "//tensorflow/c/experimental/saved_model/public:__pkg__",
    ],
    deps = [
        ":concrete_function",
        ":concrete_function_list",
        ":concrete_function_list_type",
        ":concrete_function_type",
        ":saved_model_api_type",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/experimental/saved_model/core:saved_model_api",
        "//tensorflow/c/experimental/saved_model/core:tf_saved_model_impl",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/eager:context",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "saved_model_api_type",
    hdrs = [
        "saved_model_api_type.h",
    ],
    deps = [
        "//tensorflow/c:conversion_macros",
        "//tensorflow/c/experimental/saved_model/core:saved_model_api",
    ],
)

cc_library(
    name = "tensorhandle_list",
    srcs = [
        "tensorhandle_list.cc",
    ],
    hdrs = [
        "//tensorflow/c/experimental/saved_model/public:tensorhandle_list.h",
    ],
    copts = tf_copts(),
    visibility = [
        "//tensorflow/c/experimental/saved_model/public:__pkg__",
    ],
    deps = [
        ":tensorhandle_list_type",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
    ],
)

cc_library(
    name = "tensorhandle_list_type",
    hdrs = [
        "tensorhandle_list_type.h",
    ],
    deps = [
        "//tensorflow/c:conversion_macros",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
    ],
)

tf_cc_test(
    name = "saved_model_api_test",
    size = "small",
    srcs = [
        "saved_model_api_test.cc",
    ],
    data = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/experimental/saved_model/public:saved_model_api",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
