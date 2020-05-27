# Description:
# TensorFlow Lite Support API in Java for metadata.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

android_library(
    name = "tensorflow-lite-support-metadata",
    srcs = glob(["src/java/org/tensorflow/lite/support/metadata/**/*.java"]),
    manifest = "AndroidManifest.xml",
    deps = [
        "//tensorflow/lite/experimental/support/metadata:metadata_schema_fbs_android",
        "//tensorflow/lite/experimental/support/metadata:schema_fbs_android",
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "tensorflow-lite-support-metadata-lib",
    srcs = glob(["src/java/org/tensorflow/lite/support/metadata/**/*.java"]),
    javacopts = JAVACOPTS,
    resource_jars = [
        "//tensorflow/lite/experimental/support/metadata:libmetadata_schema_java.jar",
        "//tensorflow/lite/experimental/support/metadata:libschema_fbs_java.jar",
    ],
    deps = [
        "//tensorflow/lite/experimental/support/metadata:metadata_schema_java",
        "//tensorflow/lite/experimental/support/metadata:schema_fbs_java",
        "@org_checkerframework_qual",
    ],
)
