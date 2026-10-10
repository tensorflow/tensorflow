# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

# copybara:uncomment load("@rules_python//python:proto.bzl", "py_proto_library")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "tpu_embedding_configuration_proto",
    srcs = [
        "tpu_embedding_configuration.proto",
    ],
    protodeps = [
        ":optimization_parameters_proto",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "optimization_parameters_proto",
    srcs = [
        "optimization_parameters.proto",
    ],
    protodeps = ["//xla/service:hlo_proto"],
    visibility = ["//visibility:public"],
    deps = ["@com_google_protobuf//:wrappers_proto"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "tpu_embedding_configuration_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":tpu_embedding_configuration_proto"],
# )
#
# py_proto_library(
#     name = "optimization_parameters_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":optimization_parameters_proto"],
# )
# copybara:uncomment_end
