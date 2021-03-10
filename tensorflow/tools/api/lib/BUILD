# Helper libraries for TensorFlow API compatibility test.

load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    default_visibility = [
        "//tensorflow/tools/api:__subpackages__",
        "//third_party/py/keras/api:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_proto_library(
    name = "api_objects_proto",
    srcs = ["api_objects.proto"],
)

py_library(
    name = "python_object_to_proto_visitor",
    srcs = ["python_object_to_proto_visitor.py"],
    srcs_version = "PY3",
    deps = [
        ":api_objects_proto_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "api_objects_proto_py_pb2",
#     has_services = 0,
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":api_objects_proto"],
# )
# copybara:uncomment_end
