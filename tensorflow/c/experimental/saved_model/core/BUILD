# Experimental SavedModel C APIs for TensorFlow. See RFC
# https://github.com/tensorflow/community/pull/207
# Targets in this directory are pure C++ "Classes" underlying the C API types
# under tf/c/experimental/saved_model/public/. They are subject to change and
# have visibility limited to Tensorflow's implementation only.

package(
    default_visibility = [
        "//tensorflow/c/experimental/saved_model/internal:__pkg__",
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
