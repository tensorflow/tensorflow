# Description:
# TensorFlow Lite Java API.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("//tensorflow/contrib/lite:build_def.bzl", "tflite_jni_binary")
load("//tensorflow/contrib/lite/java:aar_with_jni.bzl", "aar_with_jni")

# Building tensorflow-lite.aar including 4 variants of .so
# To build an aar for release, run below command:
# bazel build --cxxopt='--std=c++11' -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
# tensorflow/contrib/lite/java:tensorflow-lite
aar_with_jni(
    name = "tensorflow-lite",
    android_library = ":tensorflowlite",
)

android_library(
    name = "tensorflowlite",
    srcs = glob(
        [
            "src/main/java/org/tensorflow/lite/*.java",
        ],
    ),
    manifest = "AndroidManifest.xml",
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_runtime",
        "@org_checkerframework_qual",
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
        "@org_checkerframework_qual",
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
        ":libtensorflowlite_jni.so",
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
        ":libtensorflowlite_jni.so",
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
        "src/testdata/uint8.bin",
    ],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.lite.NativeInterpreterWrapperTest",
    deps = [
        ":libtensorflowlite_jni.so",
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
        "src/testdata/mobilenet.tflite.bin",
    ],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.lite.InterpreterTest",
    visibility = ["//visibility:private"],
    deps = [
        ":libtensorflowlite_jni.so",
        ":tensorflowlitelib",
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
