#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    tags = [
        "no_oss",
        "tflite_not_portable",
    ],
    deps = [
        ":util",
        "//tensorflow/contrib/lite/testing:util",
        "//tensorflow/core:lib",
        "@com_google_googletest//:gtest",
    ],
)
