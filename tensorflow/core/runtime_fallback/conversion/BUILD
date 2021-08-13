package(
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
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_tensor",
        "//tensorflow/core/tfrt/utils:error_util",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
    ],
)
