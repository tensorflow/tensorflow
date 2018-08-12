#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

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

tf_cc_test(
    name = "buffer_map_test",
    size = "small",
    srcs = ["buffer_map_test.cc"],
    tags = [
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
    name = "delegate",
    srcs = [
        "delegate.cc",
    ],
    hdrs = [
        "delegate.h",
    ],
    deps = [
        ":buffer_map",
        ":delegate_data",
        ":kernel",
        ":util",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/contrib/lite:util",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "delegate_test",
    size = "small",
    srcs = ["delegate_test.cc"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":delegate",
        ":test_util",
        "//tensorflow/contrib/lite/kernels:test_util",
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

tf_cc_test(
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
    name = "kernel",
    srcs = ["kernel.cc"],
    hdrs = ["kernel.h"],
    deps = [
        ":delegate_data",
        ":util",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/contrib/lite:string",
        "//tensorflow/contrib/lite/kernels:kernel_util",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:execute",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "kernel_test",
    size = "small",
    srcs = ["kernel_test.cc"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":delegate_data",
        ":kernel",
        ":test_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/c:c_api_internal",
        "//tensorflow/contrib/lite:string",
        "//tensorflow/contrib/lite/kernels:test_util",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":constants",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":util",
        "//tensorflow/contrib/lite:string",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)
