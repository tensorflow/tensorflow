load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_opts_nortti_if_lite_protos")
load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_cc_library", "tflite_flex_jni_library")

#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(
    default_visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/lite/android:__subpackages__",
        "//tensorflow/lite/toco/tflite:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "delegate.h",
])

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
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
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

# Define the standard flex delegate library, that pulls in the standard set
# of TensorFlow ops and kernels, using tflite_flex_cc_library with no
# portable_tensorflow_lib parameter. Custom flex delegate can be defined with
# tflite_flex_cc_library if the parameter portable_tensorflow_lib
# is provided. Ex:
# tflite_flex_cc_library(
#   name = "sample",
#   portable_tensorflow_lib = custom_portable_tensorflow_lib,
# )
tflite_flex_cc_library(
    name = "delegate",
    visibility = ["//visibility:public"],
)

# Delegate implementation that does *not* pull in the standard set of TensorFlow
# ops and kernels.
cc_library(
    name = "delegate_only_runtime",
    srcs = [
        "delegate.cc",
        "kernel.cc",
        "kernel.h",
    ],
    hdrs = [
        "delegate.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_map",
        ":delegate_data",
        ":util",
        "@flatbuffers",
        "@com_google_absl//absl/strings:strings",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:kernel_util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
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
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_map",
        "@com_google_absl//absl/memory",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core/common_runtime/eager:context",
            "//tensorflow/core/common_runtime/eager:core",
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

tf_cc_test(
    name = "kernel_test",
    size = "small",
    srcs = ["kernel_test.cc"],
    tags = ["no_gpu"],  # GPU + flex is not officially supported.
    deps = [
        ":delegate_data",
        ":delegate_only_runtime",
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
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
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

# Following targets are using for testing selective-built flex delegate
# in Java. Please don't use them for other purposes.
tflite_flex_jni_library(
    name = "test",
    models = [
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
)

java_library(
    name = "test_tensorflowlitelib_flex",
    testonly = 1,
    srcs = ["//tensorflow/lite/delegates/flex/java/src/main/java/org/tensorflow/lite/flex:flex_delegate"],
    javacopts = JAVACOPTS,
    visibility = ["//visibility:private"],
    deps = [
        ":libtensorflowlite_flex_jni.so",
        "//tensorflow/lite/java:tensorflowlitelib",
        "@org_checkerframework_qual",
    ],
)

java_test(
    name = "SelectiveBuiltInterpreterFlexTest",
    size = "small",
    srcs = [
        "//tensorflow/lite/java:portable_flex_tests",
        "//tensorflow/lite/java:portable_test_utils",
    ],
    data = [
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_cuda_on_cpu_tap",  # CUDA + flex is not officially supported.
        "no_gpu",  # GPU + flex is not officially supported.
        "no_oss",  # Currently requires --config=monolithic, b/118895218.
        # TODO(b/121204962): Re-enable test after fixing memory leaks.
        "noasan",
        "notsan",  # TODO(b/158651814) Re-enable after fixing racing condition.
    ],
    test_class = "org.tensorflow.lite.InterpreterFlexTest",
    visibility = ["//visibility:private"],
    deps = [
        ":test_tensorflowlitelib_flex",
        "//tensorflow/lite/java:tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)
