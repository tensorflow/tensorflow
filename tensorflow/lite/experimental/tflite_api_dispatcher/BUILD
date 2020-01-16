load("//tensorflow/lite:build_def.bzl", "if_tflite_experimental_runtime", "tflite_experimental_runtime_linkopts")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tflite_api_dispatcher",
    hdrs = ["tflite_api_dispatcher.h"],
    defines = if_tflite_experimental_runtime(
        if_false = [],
        if_true = ["TFLITE_EXPERIMENTAL_RUNTIME"],
    ),
    deps = [
        "//tensorflow/lite:framework",
    ] + tflite_experimental_runtime_linkopts(),
)

cc_library(
    name = "tflite_api_dispatcher_with_kernels",
    hdrs = ["tflite_api_dispatcher.h"],
    deps = [
        ":tflite_api_dispatcher",
        "//tensorflow/lite:framework",
    ] + tflite_experimental_runtime_linkopts(
        if_true = [
            # "//tensorflow/lite/experimental/tf_runtime:tfrt_tflite_interpreter_alwayslink",
            # "//third_party/tf_runtime:basic_kernels_alwayslink",
        ],
    ),
)
