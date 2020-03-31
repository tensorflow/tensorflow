load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_opts_nortti_if_lite_protos")

#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(
    default_visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/lite/toco/tflite:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "buffer_map",
    srcs = ["buffer_map.cc"],
    hdrs = ["buffer_map.h"],
    copts = tf_opts_nortti_if_lite_protos(),
    deps = [
        ":util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:ios_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/c:c_api_internal",
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
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

# Delegate implementation that pulls in the standard set of TensorFlow ops and
# kernels.
cc_library(
    name = "delegate",
    hdrs = [
        "delegate.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_data",
        ":delegate_only_runtime",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:ios_tensorflow_lib",
        ],
        "//conditions:default": [
            "//tensorflow/core:tensorflow",
            "//tensorflow/lite/c:common",
        ],
    }),
    alwayslink = 1,
)

# Delegate implementation that does *not* pull in the standard set of TensorFlow
# ops and kernels.
cc_library(
    name = "delegate_only_runtime",
    srcs = [
        "delegate.cc",
    ],
    hdrs = [
        "delegate.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_map",
        ":delegate_data",
        ":kernel",
        ":util",
        "@com_google_absl//absl/strings:strings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:ios_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:lib",
        ],
    }),
    alwayslink = 1,
)

tf_cc_test(
    name = "delegate_test",
    size = "small",
    srcs = ["delegate_test.cc"],
    tags = ["no_gpu"],  # GPU + flex is not officially supported.
    deps = [
        ":delegate",
        ":test_util",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "delegate_data",
    srcs = ["delegate_data.cc"],
    hdrs = ["delegate_data.h"],
    deps = [
        ":buffer_map",
        "@com_google_absl//absl/memory",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:ios_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core/common_runtime/eager:context",
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
        "//tensorflow/lite:framework",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/testing:util",
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
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:string",
        "//tensorflow/lite/kernels:kernel_util",
    ] + select({
        # TODO(b/111881878): The android_tensorflow_lib target pulls in the full
        # set of core TensorFlow kernels. We may want to revisit this dependency
        # to allow selective registration via build targets.
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:ios_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:execute",
            "//tensorflow/core/common_runtime/eager:tensor_handle",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:framework",
        ],
    }),
)

tf_cc_test(
    name = "kernel_test",
    size = "small",
    srcs = ["kernel_test.cc"],
    tags = ["no_gpu"],  # GPU + flex is not officially supported.
    deps = [
        ":delegate_data",
        ":kernel",
        ":test_util",
        "@com_google_googletest//:gtest",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:ios_tensorflow_lib",
        ],
        "//conditions:default": [
            "//tensorflow/core:tensorflow",
        ],
    }),
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/c:c_api_internal",
        "//tensorflow/lite:string",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:kernel_api",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:ios_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/c:c_api_internal",
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
        "//tensorflow/lite:string",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "whitelisted_flex_ops_lib",
    srcs = [
        "whitelisted_flex_ops.cc",
    ],
    hdrs = [
        "whitelisted_flex_ops.h",
    ],
)
