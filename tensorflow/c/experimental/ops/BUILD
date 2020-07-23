# Experimental ops. These will eventually be replaced by machine-generated versions.
package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "array_ops",
    srcs = [
        "array_ops.cc",
    ],
    hdrs = [
        "array_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:errors",
    ],
)
