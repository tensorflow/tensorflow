# Description:
# TensorFlow Lite Java API.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("//tensorflow/contrib/lite:build_def.bzl", "tflite_jni_binary")

android_library(
    name = "tensorflowlite",
    srcs = glob(
        [
            "src/main/java/org/tensorflow/lite/*.java",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_runtime",
        "@javax_validation",
    ],
)

android_library(
    name = "tensorflowlite_java",
    srcs = glob(
        [
            "src/main/java/org/tensorflow/lite/*.java",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "@javax_validation",
    ],
)

java_library(
    name = "tensorflowlitelib",
    srcs = glob(
        [
            "src/main/java/org/tensorflow/lite/*.java",
        ],
    ),
    javacopts = JAVACOPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":libtensorflowlite_jni.so",
        "//tensorflow/contrib/lite/java/src/main/native",
        "@javax_validation",
    ],
)

java_test(
    name = "TensorFlowLiteTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/TensorFlowLiteTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.lite.TensorFlowLiteTest",
    deps = [
        ":tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "DataTypeTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/DataTypeTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.lite.DataTypeTest",
    deps = [
        ":tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

filegroup(
    name = "libtensorflowlite_jni",
    srcs = select({
        "//conditions:default": [":libtensorflowlite_jni.so"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tflite_runtime",
    srcs = ["libtensorflowlite_jni.so"],
    visibility = ["//visibility:public"],
)

tflite_jni_binary(
    name = "libtensorflowlite_jni.so",
    deps = [
        "//tensorflow/contrib/lite/java/src/main/native",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
