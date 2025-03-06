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

load("//tensorflow/lite/experimental/litert/vendors/mediatek:mediatek_build_defs.bzl", "litert_cc_lib_with_mtk")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

litert_cc_lib_with_mtk(
    name = "neuron_adapter_api",
    srcs = [
        "neuron_adapter_api.cc",
    ],
    hdrs = [
        "neuron_adapter_api.h",
    ],
    tags = [
        # Don't build/test in OS until neuron is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        # copybara:uncomment "//third_party/neuro_pilot:latest_host_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_shared_library",
    ],
)
