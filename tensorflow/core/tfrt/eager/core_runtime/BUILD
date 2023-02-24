# Description:
# core_runtime libraries.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/tfrt/eager:__pkg__"],
    licenses = ["notice"],
)

cc_library(
    name = "core_runtime_lib",
    srcs = [
        "op_handler_registry.cc",
        "op_handler_selector.cc",
    ],
    hdrs = [
        "op_handler_registry.h",
        "op_handler_selector.h",
    ],
    deps = [
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime/eager:attr_builder",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:placement_utils",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:support",
    ],
)
