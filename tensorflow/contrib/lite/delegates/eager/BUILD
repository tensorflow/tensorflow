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
        "//tensorflow/contrib/lite:kernel_api",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
        ],
    }),
)

tf_cc_test(
    name = "buffer_map_test",
    size = "small",
    srcs = ["buffer_map_test.cc"],
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
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/contrib/lite:util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core:lib",
        ],
    }),
)

tf_cc_test(
    name = "delegate_test",
    size = "small",
    srcs = ["delegate_test.cc"],
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
        "//tensorflow/core/common_runtime/eager:context",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:lib",
        ],
    }),
)

tf_cc_test(
    name = "delegate_data_test",
    size = "small",
    srcs = ["delegate_data_test.cc"],
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
        "@flatbuffers",
        "//tensorflow/contrib/lite:kernel_api",
        "//tensorflow/contrib/lite:string",
        "//tensorflow/contrib/lite/kernels:kernel_util",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:execute",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
    ] + select({
        # TODO(b/111881878): The android_tensorflow_lib target pulls in the full
        # set of core TensorFlow kernels. We may want to revisit this dependency
        # to allow selective registration via build targets.
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib",
        ],
        "//conditions:default": [
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:framework",
            "//tensorflow/core:tensorflow",
        ],
    }),
)

tf_cc_test(
    name = "kernel_test",
    size = "small",
    srcs = ["kernel_test.cc"],
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
        "//tensorflow/c:c_api_internal",
        "//tensorflow/contrib/lite:kernel_api",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite_no_runtime",
        ],
        "//conditions:default": [
            "//tensorflow/core:lib",
            "//tensorflow/core:framework",
        ],
    }),
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//tensorflow/contrib/lite:string",
        "//tensorflow/contrib/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)
