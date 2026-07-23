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
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable", "get_compatible_with_portable")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "compile_options_proto",
    srcs = ["compile_options.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:xla_data_proto",
        "//xla:xla_proto",
        "//xla/stream_executor:device_description_proto",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "compile_options_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":compile_options_proto"],
# )
# copybara:uncomment_end

tf_proto_library(
    name = "execute_options_proto",
    srcs = ["execute_options.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "executable_metadata_proto",
    srcs = ["executable_metadata.proto"],
    protodeps = ["@com_google_protobuf//:any_proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "pjrt_abi_version_proto",
    srcs = ["pjrt_abi_version.proto"],
    protodeps = [],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "pjrt_device_dimensions_proto",
    srcs = ["pjrt_device_dimensions.proto"],
    compatible_with = (get_compatible_with_libtpu_portable() + get_compatible_with_portable()),
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "pjrt_value_type_proto",
    srcs = ["pjrt_value_type.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "pjrt_partial_program_proto",
    srcs = ["pjrt_partial_program.proto"],
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "topology_description_proto",
    srcs = ["topology_description.proto"],
    compatible_with = (get_compatible_with_libtpu_portable() + get_compatible_with_portable()),
    visibility = ["//visibility:public"],
    deps = ["@com_google_protobuf//:any_proto"],
)
