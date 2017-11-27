# Helper libraries for TensorFlow API compatibility test.

package(
    default_visibility = ["//tensorflow/tools/api:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "api_objects_proto",
    srcs = ["api_objects.proto"],
)

py_library(
    name = "python_object_to_proto_visitor",
    srcs = ["python_object_to_proto_visitor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":api_objects_proto_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
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
