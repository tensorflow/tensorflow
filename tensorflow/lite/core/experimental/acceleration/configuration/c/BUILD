load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "stable_delegate",
    hdrs = ["stable_delegate.h"],
    tflite_deps = [
        "//tensorflow/lite/core/shims:delegate_plugin",
    ],
    visibility = ["//visibility:public"],
)
