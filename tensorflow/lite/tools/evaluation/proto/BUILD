# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library_py",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "evaluation_stages_proto",
    srcs = [
        "evaluation_stages.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":preprocessing_steps_proto"],
)

cc_proto_library(
    name = "evaluation_stages_cc_proto",
    deps = ["evaluation_stages_proto"],
)

java_proto_library(
    name = "evaluation_stages_java_proto",
    deps = ["evaluation_stages_proto"],
)

tf_proto_library_py(
    name = "evaluation_stages",  # bzl adds _py
    srcs = [
        "evaluation_stages.proto",
    ],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "evaluation_config_proto",
    srcs = [
        "evaluation_config.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":evaluation_stages_proto"],
)

cc_proto_library(
    name = "evaluation_config_cc_proto",
    deps = ["evaluation_config_proto"],
)

java_proto_library(
    name = "evaluation_config_java_proto",
    deps = ["evaluation_config_proto"],
)

proto_library(
    name = "preprocessing_steps_proto",
    srcs = [
        "preprocessing_steps.proto",
    ],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "preprocessing_steps_cc_proto",
    deps = ["preprocessing_steps_proto"],
)

java_proto_library(
    name = "preprocessing_steps_java_proto",
    deps = ["preprocessing_steps_proto"],
)
