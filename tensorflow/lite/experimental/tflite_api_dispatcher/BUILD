load("//tensorflow/lite:build_def.bzl", "if_tflite_experimental_runtime", "tflite_experimental_runtime_linkopts")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tflite_api_dispatcher",
    hdrs = ["tflite_api_dispatcher.h"],
    defines = if_tflite_experimental_runtime(
        if_eager = ["TFLITE_EXPERIMENTAL_RUNTIME_EAGER"],
        if_non_eager = ["TFLITE_EXPERIMENTAL_RUNTIME_NON_EAGER"],
        if_none = [],
    ),
    deps = [
        "//tensorflow/lite:framework_lib",
    ] + tflite_experimental_runtime_linkopts(),
)

cc_library(
    name = "tflite_api_dispatcher_with_kernels",
    hdrs = ["tflite_api_dispatcher.h"],
    deps = [
        ":tflite_api_dispatcher",
        "//tensorflow/lite:framework_lib",
    ] + tflite_experimental_runtime_linkopts(
        if_eager = [
            # "//tensorflow/lite/experimental/tf_runtime/opdef:tflrt_opdefs",
            # "//tensorflow/lite/experimental/tf_runtime/tfrt_ops:tfrt_tflite_ops_alwayslink",
        ],
        if_non_eager = [
            # "//tensorflow/lite/experimental/tf_runtime/tfrt_kernels:tfrt_tflite_interpreter_alwayslink",
            # "//third_party/tf_runtime:basic_kernels_alwayslink",
        ],
    ),
)
