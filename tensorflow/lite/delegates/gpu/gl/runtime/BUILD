package(default_visibility = ["//tensorflow/lite/delegates/gpu/gl:__subpackages__"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "shared_buffer",
    hdrs = ["shared_buffer.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "//tensorflow/lite/delegates/gpu/gl:gl_call",
        "//tensorflow/lite/delegates/gpu/gl:object",
        "//tensorflow/lite/delegates/gpu/gl:portable",
    ],
)
