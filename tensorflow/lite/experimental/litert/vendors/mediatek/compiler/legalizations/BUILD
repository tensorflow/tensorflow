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
    default_visibility = ["//tensorflow/lite/experimental/litert/vendors/mediatek/compiler:__subpackages__"],
)

cc_library(
    name = "operand_map",
    srcs = ["operand_map.cc"],
    hdrs = ["operand_map.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_element_type",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "add_op_legalization",
    srcs = ["add_op_legalization.cc"],
    hdrs = ["add_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)
