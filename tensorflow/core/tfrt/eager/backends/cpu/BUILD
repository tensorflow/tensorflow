package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/core/tfrt/eager/...",
    ],
)

cc_library(
    name = "cpu_registration_alwayslink",
    srcs = [
        "cpu_registration.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/tfrt/eager:core_runtime",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
    ],
    alwayslink = 1,
)
