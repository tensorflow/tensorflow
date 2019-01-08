# Description:
# OVIC Benchmarker Java API.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")

# Build targets for OVIC classification.
java_test(
    name = "OvicClassifierTest",
    size = "medium",
    srcs = ["src/test/java/org/tensorflow/ovic/OvicClassifierTest.java"],
    data = [
        "//tensorflow/lite/java/ovic/src/testdata:labels.txt",
        "//tensorflow/lite/java/ovic/src/testdata:ovic_testdata",
    ],
    javacopts = JAVACOPTS,
    tags = ["no_oss"],
    test_class = "org.tensorflow.ovic.OvicClassifierTest",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/java/ovic:ovicbenchmarkerlib_java",
        "@com_google_truth",
        "@junit",
    ],
)

java_binary(
    name = "ovic_validator",
    srcs = ["src/main/java/org/tensorflow/ovic/OvicValidator.java"],
    data = [
        "//tensorflow/lite/java/ovic/src/testdata:labels.txt",
    ],
    main_class = "org.tensorflow.ovic.OvicValidator",
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite/java/ovic:ovicbenchmarkerlib_java",
        "//tensorflow/lite/java/ovic:ovicdetectionbenchmarkerlib_java",
    ],
)

android_library(
    name = "ovicbenchmarkerlib",
    srcs = [
        "src/main/java/org/tensorflow/ovic/OvicBenchmarker.java",
        "src/main/java/org/tensorflow/ovic/OvicClassificationResult.java",
        "src/main/java/org/tensorflow/ovic/OvicClassifier.java",
        "src/main/java/org/tensorflow/ovic/OvicClassifierBenchmarker.java",
    ],
    manifest = "//tensorflow/lite/java:AndroidManifest.xml",
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite/java:tensorflowlite",
        "//tensorflow/lite/java/src/testhelper/java/org/tensorflow/lite:testhelper",
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "ovicbenchmarkerlib_java",
    srcs = [
        "src/main/java/org/tensorflow/ovic/OvicClassificationResult.java",
        "src/main/java/org/tensorflow/ovic/OvicClassifier.java",
    ],
    javacopts = JAVACOPTS,
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite/java:libtensorflowlite_jni.so",
        "//tensorflow/lite/java:tensorflowlite_java",
        "//tensorflow/lite/java/src/main/native",
        "//tensorflow/lite/java/src/testhelper/java/org/tensorflow/lite:testhelper",
        "@org_checkerframework_qual",
    ],
)

# Build targets for OVIC detection.
java_test(
    name = "OvicDetectorTest",
    size = "medium",
    srcs = ["src/test/java/org/tensorflow/ovic/OvicDetectorTest.java"],
    data = [
        "//tensorflow/lite/java/ovic/src/testdata:coco_labels.txt",
        "//tensorflow/lite/java/ovic/src/testdata:ovic_testdata",
    ],
    javacopts = JAVACOPTS,
    tags = ["no_oss"],
    test_class = "org.tensorflow.ovic.OvicDetectorTest",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/java/ovic:ovicdetectionbenchmarkerlib_java",
        "@com_google_truth",
        "@junit",
    ],
)

android_library(
    name = "ovicdetectionbenchmarkerlib",
    srcs = [
        "src/main/java/org/tensorflow/ovic/BoundingBox.java",
        "src/main/java/org/tensorflow/ovic/OvicBenchmarker.java",
        "src/main/java/org/tensorflow/ovic/OvicDetectionResult.java",
        "src/main/java/org/tensorflow/ovic/OvicDetector.java",
        "src/main/java/org/tensorflow/ovic/OvicDetectorBenchmarker.java",
    ],
    manifest = "//tensorflow/lite/java:AndroidManifest.xml",
    deps = [
        "//tensorflow/lite/java:tensorflowlite",
        "//tensorflow/lite/java/src/testhelper/java/org/tensorflow/lite:testhelper",
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "ovicdetectionbenchmarkerlib_java",
    srcs = [
        "src/main/java/org/tensorflow/ovic/BoundingBox.java",
        "src/main/java/org/tensorflow/ovic/OvicDetectionResult.java",
        "src/main/java/org/tensorflow/ovic/OvicDetector.java",
    ],
    javacopts = JAVACOPTS,
    deps = [
        "//tensorflow/lite/java:libtensorflowlite_jni.so",
        "//tensorflow/lite/java:tensorflowlite_java",
        "//tensorflow/lite/java/src/main/native",
        "//tensorflow/lite/java/src/testhelper/java/org/tensorflow/lite:testhelper",
        "@org_checkerframework_qual",
    ],
)
