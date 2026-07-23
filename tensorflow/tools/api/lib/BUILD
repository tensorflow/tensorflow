# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Helper libraries for TensorFlow API compatibility test.

load("@rules_python//python:proto.bzl", "py_proto_library")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")
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

py_library(
    name = "python_object_to_proto_visitor",
    srcs = ["python_object_to_proto_visitor.py"],
    strict_deps = True,
    deps = [
        ":api_objects_proto_py",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_test(
    name = "python_object_to_proto_visitor_test",
    srcs = ["python_object_to_proto_visitor_test.py"],
    strict_deps = True,
    deps = [
        ":api_objects_proto_py",
        ":python_object_to_proto_visitor",
        "//tensorflow/python/platform:test",
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
