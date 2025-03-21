# Copyright (c) 2025 MediaTek Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

flatbuffer_cc_library(
    name = "neuron_litert_schema",
    srcs = ["neuron_schema.fbs"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "mediatek_litert_schema",
    hdrs = [
        "schema_resolver.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "neuron_litert_schema",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers//:runtime_cc",
    ],
)
