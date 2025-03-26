package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/runtime_fallback:internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/runtime_fallback/conversion/...",
    ],
)

cc_library(
    name = "conversion_alwayslink",
    srcs = [
        "static_registration.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":conversion_lib",
        "@tf_runtime//:tensor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "conversion_lib",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//tensorflow/core/runtime_fallback/kernel:tensor_util",
        "//tensorflow/core/runtime_fallback/runtime:kernel_utils",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_tensor",
        "@com_google_absl//absl/status",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
    ],
)
