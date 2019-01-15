# Description:
# TensorFlow Lite Java API.

package(default_visibility = [
    "//tensorflow/lite/java/ovic:__pkg__",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")

JAVA_SRCS = glob([
    "src/main/java/org/tensorflow/lite/*.java",
])

# Building tensorflow-lite.aar including 4 variants of .so
# To build an aar for release, run below command:
# bazel build --cxxopt='--std=c++11' -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
# tensorflow/lite/java:tensorflow-lite
aar_with_jni(
    name = "tensorflow-lite",
    android_library = ":tensorflowlite",
)

# EXPERIMENTAL: AAR target that supports TensorFlow op execution with TFLite.
aar_with_jni(
    name = "tensorflow-lite-with-select-tf-ops",
    android_library = ":tensorflowlite_flex",
)

android_library(
    name = "tensorflowlite",
    srcs = JAVA_SRCS,
    manifest = "AndroidManifest.xml",
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflowlite_native",
        "@org_checkerframework_qual",
    ],
)

# EXPERIMENTAL: Android target that supports TensorFlow op execution with TFLite.
android_library(
    name = "tensorflowlite_flex",
    srcs = JAVA_SRCS,
    manifest = "AndroidManifest.xml",
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflowlite_native_flex",
        "@org_checkerframework_qual",
    ],
)

android_library(
    name = "tensorflowlite_java",
    srcs = JAVA_SRCS,
    visibility = ["//visibility:public"],
    deps = [
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "tensorflowlitelib",
    srcs = JAVA_SRCS,
    javacopts = JAVACOPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":libtensorflowlite_jni.so",
        "@org_checkerframework_qual",
    ],
)

# EXPERIMENTAL: Java target that supports TensorFlow op execution with TFLite.
java_library(
    name = "tensorflowlitelib_flex",
    srcs = JAVA_SRCS,
    javacopts = JAVACOPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":libtensorflowlite_flex_jni.so",
        "@org_checkerframework_qual",
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

java_test(
    name = "NativeInterpreterWrapperTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/NativeInterpreterWrapperTest.java"],
    data = [
        "src/testdata/add.bin",
        "src/testdata/int32.bin",
        "src/testdata/int64.bin",
        "src/testdata/invalid_model.bin",
        "src/testdata/quantized.bin",
        "src/testdata/string.bin",
        "src/testdata/uint8.bin",
        "src/testdata/with_custom_op.lite",
    ],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.lite.NativeInterpreterWrapperTest",
    deps = [
        ":tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

# TODO: generate large models at runtime, instead of storing them.
java_test(
    name = "InterpreterTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/InterpreterTest.java"],
    data = [
        "src/testdata/add.bin",
        "//tensorflow/lite:testdata/multi_add.bin",
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.lite.InterpreterTest",
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflowlitelib",
        "//tensorflow/lite/java/src/test/native:libtensorflowlite_test_jni.so",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "InterpreterFlexTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/InterpreterFlexTest.java"],
    data = [
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_oss",  # Currently requires --config=monolithic, b/118895218.
    ],
    test_class = "org.tensorflow.lite.InterpreterFlexTest",
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflowlitelib_flex",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "TensorTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/TensorTest.java"],
    data = [
        "src/testdata/add.bin",
    ],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.lite.TensorTest",
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
    name = "tensorflowlite_native",
    srcs = ["libtensorflowlite_jni.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorflowlite_native_flex",
    srcs = ["libtensorflowlite_flex_jni.so"],
    visibility = ["//visibility:public"],
)

tflite_jni_binary(
    name = "libtensorflowlite_jni.so",
    deps = [
        "//tensorflow/lite/java/src/main/native",
    ],
)

# EXPERIMENTAL: Native target that supports TensorFlow op execution with TFLite.
tflite_jni_binary(
    name = "libtensorflowlite_flex_jni.so",
    deps = [
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/java/src/main/native",
        "//tensorflow/lite/java/src/main/native:init_tensorflow",
    ],
)
