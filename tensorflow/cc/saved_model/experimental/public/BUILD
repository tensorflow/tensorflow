# Experimental C++ SavedModel Header Only APIs. See RFC
# https://github.com/tensorflow/community/pull/207

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # This is intentionally public
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "concrete_function",
    hdrs = [
        "concrete_function.h",
    ],
    deps = [
        ":function_metadata",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/experimental/saved_model/public:concrete_function",
        "//tensorflow/cc/experimental/base/public:status",
    ],
)

cc_library(
    name = "concrete_function_list",
    hdrs = [
        "concrete_function_list.h",
    ],
    deps = [
        ":concrete_function",
        "//tensorflow/c/experimental/saved_model/public:concrete_function_list",
    ],
)

cc_library(
    name = "function_metadata",
    hdrs = [
        "function_metadata.h",
    ],
    deps = [
        "//tensorflow/c/experimental/saved_model/public:function_metadata",
    ],
)

cc_library(
    name = "saved_model_api",
    hdrs = [
        "saved_model_api.h",
    ],
    deps = [
        ":concrete_function",
        ":concrete_function_list",
        ":signature_def_function",
        "//tensorflow/c/experimental/saved_model/public:saved_model_api",
        "//tensorflow/cc/experimental/base/public:runtime",
        "//tensorflow/cc/experimental/base/public:status",
    ],
)

cc_library(
    name = "signature_def_function",
    hdrs = [
        "signature_def_function.h",
    ],
    deps = [
        ":signature_def_function_metadata",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/experimental/saved_model/public:signature_def_function",
        "//tensorflow/cc/experimental/base/public:status",
    ],
)

cc_library(
    name = "signature_def_function_metadata",
    hdrs = [
        "signature_def_function_metadata.h",
    ],
    deps = [
        "//tensorflow/c/experimental/saved_model/public:signature_def_function_metadata",
    ],
)
