load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/runtime_fallback/...",
        "//tensorflow/compiler/mlir/tfrt/tests/tfrt_fallback/...",
    ] + if_google([
        "//learning/brain/experimental/mlir/tflite/tfmrt/...",
        "//learning/brain/experimental/tfrt/...",
        "//learning/brain/tfrt/...",
        "//learning/brain/mobile/lite/...",
    ]),
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

# This build target contains fallback kernels only. Some of the native TFRT
# ops/kernels (e.g. eigen based matmul) can be expensive to build, but they are
# not needed for fallback testing.
tf_cc_binary(
    name = "tf_bef_executor",
    testonly = True,
    tags = ["no_oss"],
    deps = [
        ":bef_executor_lib",
        "//tensorflow/core/platform:stream_executor",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_kernels_alwayslink",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_op_handler",
        # copybara:uncomment "//tensorflow/core/runtime_fallback/test:forwarding_test_kernels",
        # copybara:uncomment "//tensorflow/core/runtime_fallback/test:tfrt_forwarding_kernels_alwayslink",
        "//tensorflow/core/runtime_fallback/conversion:conversion_alwayslink",
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt_kernels_alwayslink",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:tensor_alwayslink",
        "@tf_runtime//:test_kernels_alwayslink",
        # copybara:uncomment "@tf_runtime//backends/cpu:proto_alwayslink",
        # copybara:uncomment "@tf_runtime//backends/cpu:image_alwayslink",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        "@tf_runtime//backends/cpu:test_ops_alwayslink",
        "@tf_runtime//backends/jitrt:jitrt_corert_kernels_alwayslink",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:all_kernels",
        ],
    }) + if_cuda([
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_gpu_alwayslink",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_gpu_alwayslink",
    ]) + select({
        "@tf_runtime//:gpu_enabled": [
            "@tf_runtime//backends/gpu:gpu_op_handler_alwayslink",
            "@tf_runtime//backends/gpu:gpu_test_ops_alwayslink",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "bef_executor_flags",
    testonly = True,
    srcs = ["bef_executor_flags.cc"],
    hdrs = ["bef_executor_flags.h"],
    visibility = ["//third_party/tf_runtime_google:__pkg__"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@tf_runtime//:bef_executor_driver",
    ],
)

cc_library(
    name = "bef_executor_lib",
    testonly = True,
    srcs = [
        "tf_bef_executor_main.cc",
    ],
    tags = ["no_oss"],
    visibility = [
        ":internal",
        "//tensorflow/core/tfrt/eager:__subpackages__",
    ],
    deps = if_google([
        "//tensorflow/core/runtime_fallback/test:test_tf_opkernels_alwayslink",
        "//tensorflow/core/runtime_fallback/test:test_tfrt_kernels_alwayslink",
        "//third_party/tf_runtime_google:xprof_tracing_sink_alwayslink",
    ]) + [
        ":bef_executor_flags",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "//tensorflow/core/tfrt/utils:thread_pool",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/runtime_fallback/util:fallback_test_util",
        "@tf_runtime//:bef_executor_driver",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:io_alwayslink",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core/platform:platform_port",
        ],
    }),
)
