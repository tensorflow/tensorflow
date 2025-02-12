# Helper libraries for TensorFlow API compatibility test.

load("//tensorflow:strict.default.bzl", "py_strict_library")

# Placeholder: load py_proto_library
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/tools/api:__subpackages__",
        "//third_party/py/tf_keras/api:__subpackages__",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "api_objects_proto",
    srcs = ["api_objects.proto"],
)

py_strict_library(
    name = "python_object_to_proto_visitor",
    srcs = ["python_object_to_proto_visitor.py"],
    deps = [
        ":api_objects_proto_py",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_inspect",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "api_objects_proto_py_pb2",
#     has_services = 0,
#     visibility = ["//visibility:public"],
#     deps = [":api_objects_proto"],
# )
# copybara:uncomment_end
