load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/runtime_fallback:internal",
        "//tensorflow/core/tfrt/saved_model:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "gpurt_kernels",
    srcs = ["gpurt_kernels.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:copy_tensor",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_utils",
        "//tensorflow/core/runtime_fallback/kernel:tensor_util",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "//tensorflow/core/tfrt/utils:gpu_variables_table",
        "//tensorflow/core/tfrt/utils:tensor_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
    ],
    alwayslink = True,
)
