# Description:
# OVIC Benchmarker Java API.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")

java_test(
    name = "OvicClassifierTest",
    size = "medium",
    srcs = ["src/test/java/org/tensorflow/ovic/OvicClassifierTest.java"],
    data = [
        "//tensorflow/contrib/lite/java/ovic/src/testdata:labels.txt",
        "//tensorflow/contrib/lite/java/ovic/src/testdata:ovic_testdata",
    ],
    javacopts = JAVACOPTS,
    tags = ["no_oss"],
    test_class = "org.tensorflow.ovic.OvicClassifierTest",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/lite/java/ovic:ovicbenchmarkerlib_java",
        "@com_google_truth",
        "@junit",
    ],
)

java_binary(
    name = "ovic_validator",
    srcs = ["src/main/java/org/tensorflow/ovic/OvicValidator.java"],
    data = [
        "//tensorflow/contrib/lite/java/ovic/src/testdata:labels.txt",
    ],
    main_class = "org.tensorflow.ovic.OvicValidator",
    tags = ["no_oss"],
    deps = [
        "//tensorflow/contrib/lite/java/ovic:ovicbenchmarkerlib_java",
    ],
)

android_library(
    name = "ovicbenchmarkerlib",
    srcs = [
        "src/main/java/org/tensorflow/ovic/OvicClassifier.java",
        "src/main/java/org/tensorflow/ovic/OvicSingleImageResult.java",
    ],
    manifest = "//tensorflow/contrib/lite/java:AndroidManifest.xml",
    tags = ["no_oss"],
    deps = [
        "//tensorflow/contrib/lite/java:tensorflowlite",
        "//tensorflow/contrib/lite/java/src/testhelper/java/org/tensorflow/lite:testhelper",
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "ovicbenchmarkerlib_java",
    srcs = [
        "src/main/java/org/tensorflow/ovic/OvicClassifier.java",
        "src/main/java/org/tensorflow/ovic/OvicSingleImageResult.java",
    ],
    javacopts = JAVACOPTS,
    tags = ["no_oss"],
    deps = [
        "//tensorflow/contrib/lite/java:libtensorflowlite_jni.so",
        "//tensorflow/contrib/lite/java:tensorflowlite_java",
        "//tensorflow/contrib/lite/java/src/main/native",
        "//tensorflow/contrib/lite/java/src/testhelper/java/org/tensorflow/lite:testhelper",
        "@org_checkerframework_qual",
    ],
)
