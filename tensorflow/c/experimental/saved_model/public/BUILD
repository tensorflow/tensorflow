# Experimental SavedModel C APIs for TensorFlow.
# See RFC https://github.com/tensorflow/community/pull/207
# All headers are on the public surface of Tensorflow's C API.
# Once moved out of experimental, these will be stable.
# The idea behind a separate public/ directory is to make apparent
# which headers are part of TF's public interface (and which headers)
# are implementation details. This structure allows us to also perform future
# programmatic checks that all "public" headers only include other "public"
# headers.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # This is intentionally public
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# TODO(bmzhao): Remove these exports_files and rules, swap with cc_public_library instead.
# cc_public_library would allows us to separate the header dep graph from header+srcs dep graph.
exports_files(
    [
        "concrete_function.h",
        "concrete_function_list.h",
        "function_metadata.h",
        "saved_model_api.h",
        "signature_def_function.h",
        "signature_def_function_metadata.h",
        "signature_def_param.h",
        "signature_def_param_list.h",
        "tensor_spec.h",
    ],
    visibility = ["//tensorflow/c/experimental/saved_model/internal:__pkg__"],
)

# The purpose of this header is to provide insulation against
# future changes where we rename/move a public header, without
# forcing all clients to change their "#includes".
cc_library(
    name = "c_saved_model_api",
    hdrs = ["c_saved_model_api.h"],
    deps = [
        ":concrete_function",
        ":concrete_function_list",
        ":function_metadata",
        ":saved_model_api",
        ":signature_def_function",
        ":signature_def_function_metadata",
        ":signature_def_param",
        ":signature_def_param_list",
        ":tensor_spec",
    ],
)

alias(
    name = "concrete_function",
    actual = "//tensorflow/c/experimental/saved_model/internal:concrete_function",
)

alias(
    name = "concrete_function_list",
    actual = "//tensorflow/c/experimental/saved_model/internal:concrete_function_list",
)

alias(
    name = "function_metadata",
    actual = "//tensorflow/c/experimental/saved_model/internal:function_metadata",
)

alias(
    name = "saved_model_api",
    actual = "//tensorflow/c/experimental/saved_model/internal:saved_model_api",
)

alias(
    name = "signature_def_function",
    actual = "//tensorflow/c/experimental/saved_model/internal:signature_def_function",
)

alias(
    name = "signature_def_function_metadata",
    actual = "//tensorflow/c/experimental/saved_model/internal:signature_def_function_metadata",
)

alias(
    name = "signature_def_param",
    actual = "//tensorflow/c/experimental/saved_model/internal:signature_def_param",
)

alias(
    name = "signature_def_param_list",
    actual = "//tensorflow/c/experimental/saved_model/internal:signature_def_param_list",
)

alias(
    name = "tensor_spec",
    actual = "//tensorflow/c/experimental/saved_model/internal:tensor_spec",
)
