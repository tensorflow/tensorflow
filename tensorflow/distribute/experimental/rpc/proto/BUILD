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

load("@rules_python//python:proto.bzl", "py_proto_library")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/distribute/experimental/rpc:__subpackages__",
        "//tensorflow/python/distribute/experimental/rpc:__subpackages__",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "tf_rpc_service_proto",
    srcs = ["tf_rpc_service.proto"],
    has_services = 1,
    create_go_proto = False,
    create_grpc_library = True,
    create_java_proto = False,
    create_service = True,
    protodeps = tf_additional_all_protos(),
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "tf_rpc_service_py_pb2",
#     deps = [":tf_rpc_service_proto"],
# )
# copybara:uncomment_end
