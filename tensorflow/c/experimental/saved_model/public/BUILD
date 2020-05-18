# Experimental SavedModel C APIs for TensorFlow.
# See RFC https://github.com/tensorflow/community/pull/207
# All headers are on the public surface of Tensorflow's C API.
# Once moved out of experimental, these will be stable.
# The idea behind a separate public/ directory is to make apparent
# which headers are part of TF's public interface (and which headers)
# are implementation details. This structure allows us to also perform future
# programmatic checks that all "public" headers only include other "public"
# headers.

package(
    # This is intentionally public
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(bmzhao): Remove these exports_files and rules, swap with cc_public_library instead.
# cc_public_library would allows us to separate the header dep graph from header+srcs dep graph.
exports_files(
    [
        "concrete_function.h",
        "concrete_function_list.h",
        "function_metadata.h",
        "saved_model_api.h",
        "tensorhandle_list.h",
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
        ":tensorhandle_list",
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
    name = "tensorhandle_list",
    actual = "//tensorflow/c/experimental/saved_model/internal:tensorhandle_list",
)
