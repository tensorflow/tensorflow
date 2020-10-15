load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tflite_api_dispatcher",
    hdrs = ["tflite_api_dispatcher.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:framework_lib",
    ],
)

cc_library(
    name = "tflite_api_dispatcher_with_kernels",
    hdrs = ["tflite_api_dispatcher.h"],
    deps = [
        ":tflite_api_dispatcher",
        "//tensorflow/lite:framework_lib",
    ],
)
