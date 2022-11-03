load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "vendor_delegate",
    hdrs = ["vendor_delegate.h"],
    tflite_deps = [
        "//tensorflow/lite/core/shims:delegate_plugin",
    ],
    visibility = ["//visibility:public"],
)
