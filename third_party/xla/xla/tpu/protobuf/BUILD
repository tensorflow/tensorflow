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
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
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

tf_proto_library(
    name = "compilation_result_proto",
    srcs = [
        "compilation_result.proto",
    ],
    protodeps = [
        "//xla:xla_data_proto",
        "//xla/service:hlo_proto",
        "//xla/tsl/protobuf:error_codes_proto_impl",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tpu_embedding_optimization_parameters_utils",
    srcs = ["tpu_embedding_optimization_parameters_utils.cc"],
    hdrs = ["tpu_embedding_optimization_parameters_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":optimization_parameters_proto_cc",
        "//xla:xla_data_proto_cc",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "tpu_embedding_configuration_utils",
    srcs = ["tpu_embedding_configuration_utils.cc"],
    hdrs = ["tpu_embedding_configuration_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":optimization_parameters_proto_cc",
        ":tpu_embedding_configuration_proto_cc",
        ":tpu_embedding_optimization_parameters_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
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
#
# py_proto_library(
#     name = "compilation_result_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":compilation_result_proto"],
# )
# copybara:uncomment_end
