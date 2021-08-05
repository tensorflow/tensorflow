load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
)

package(
    default_visibility = [":friends"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/core/tfrt/eager/...",
    ],
)

tf_cuda_library(
    name = "gpu_registration_alwayslink",
    srcs = [
        "gpu_registration.cc",
    ],
    compatible_with = [],
    tags = [
        "no_oss",
        "requires_cuda",
    ],  # Only build this library with --config=cuda.
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_gpu_alwayslink",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_gpu_alwayslink",
        "//tensorflow/core/tfrt/eager:core_runtime",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:support",
        "@tf_runtime//backends/gpu:gpu_op_handler",
    ],
    alwayslink = 1,
)
