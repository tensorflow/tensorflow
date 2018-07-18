#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "buffer_map",
    srcs = ["buffer_map.cc"],
    hdrs = ["buffer_map.h"],
    deps = [
        ":util",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "buffer_map_test",
    size = "small",
    srcs = ["buffer_map_test.cc"],
    tags = [
        "no_oss",
        "tflite_not_portable",
    ],
    deps = [
        ":buffer_map",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:util",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "delegate_data",
    srcs = ["delegate_data.cc"],
    hdrs = ["delegate_data.h"],
    deps = [
        ":buffer_map",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/eager:context",
    ],
)

cc_test(
    name = "delegate_data_test",
    size = "small",
    srcs = ["delegate_data_test.cc"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":delegate_data",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:util",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/c:c_api_internal",
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
