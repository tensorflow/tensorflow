# Copyright 2024 Google LLC.
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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "flatbuffer_tools",
    srcs = ["flatbuffer_tools.cc"],
    hdrs = [
        "flatbuffer_tools.h",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
    ],
)
