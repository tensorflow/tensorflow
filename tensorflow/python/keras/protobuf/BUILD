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

# Description:
#   Contains Keras protobufs

load("@rules_python//python:proto.bzl", "py_proto_library")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "projector_config_proto",
    srcs = ["projector_config.proto"],
)

tf_proto_library(
    name = "saved_metadata_proto",
    srcs = ["saved_metadata.proto"],
    protodeps = [":versions_proto"],
)

tf_proto_library(
    name = "versions_proto",
    srcs = ["versions.proto"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "saved_metadata_proto_py_pb2",
#     deps = [":saved_metadata_proto"],
# )
#
# py_proto_library(
#     name = "projector_config_proto_py_pb2",
#     deps = [":projector_config_proto"],
# )
#
# py_proto_library(
#     name = "versions_proto_py_pb2",
#     deps = [":versions_proto"],
# )
# copybara:uncomment_end
