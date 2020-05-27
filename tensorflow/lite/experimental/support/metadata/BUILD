load("//tensorflow:tensorflow.bzl", "py_test")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_android_library", "flatbuffer_cc_library", "flatbuffer_java_library", "flatbuffer_py_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["metadata_schema.fbs"])

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//tensorflow/lite/schema:schema.fbs"],
)

# Generic schema for inference on device.
flatbuffer_android_library(
    name = "schema_fbs_android",
    srcs = ["//tensorflow/lite/schema:schema.fbs"],
    custom_package = "org.tensorflow.lite.schema",
)

flatbuffer_java_library(
    name = "schema_fbs_java",
    srcs = ["//tensorflow/lite/schema:schema.fbs"],
    custom_package = "org.tensorflow.lite.schema",
)

# Generic schema for model metadata.
flatbuffer_cc_library(
    name = "metadata_schema_cc",
    srcs = ["metadata_schema.fbs"],
)

flatbuffer_py_library(
    name = "metadata_schema_py",
    srcs = ["metadata_schema.fbs"],
)

flatbuffer_java_library(
    name = "metadata_schema_java",
    srcs = ["metadata_schema.fbs"],
    custom_package = "org.tensorflow.lite.support.metadata.schema",
)

flatbuffer_android_library(
    name = "metadata_schema_fbs_android",
    srcs = ["metadata_schema.fbs"],
    custom_package = "org.tensorflow.lite.support.metadata.schema",
)

py_library(
    name = "metadata",
    srcs = ["metadata.py"],
    data = [
        "//tensorflow/lite/experimental/support/metadata:metadata_schema.fbs",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_schema_py",
        ":schema_py",
        "//tensorflow/lite/experimental/support/metadata/cc/python:_pywrap_metadata_version",
        "//tensorflow/lite/experimental/support/metadata/flatbuffers_lib:_pywrap_flatbuffers",
        "//tensorflow/python:platform",
        "@flatbuffers//:runtime_py",
    ],
)

py_test(
    name = "metadata_test",
    srcs = ["metadata_test.py"],
    data = ["testdata/golden_json.json"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_mac",  # TODO(b/148247402): flatbuffers import broken on Mac OS.
    ],
    deps = [
        ":metadata",
        ":metadata_schema_py",
        ":schema_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "@flatbuffers//:runtime_py",
        "@six_archive//:six",
    ],
)
