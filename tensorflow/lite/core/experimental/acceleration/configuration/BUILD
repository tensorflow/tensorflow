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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite/core:special_rules.bzl", "delegate_registry_visibility_allowlist")
load("//tensorflow/lite/core/c:special_rules.bzl", "experimental_acceleration_api_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    licenses = ["notice"],
)

cc_library(
    name = "delegate_registry",
    hdrs = ["delegate_registry.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = delegate_registry_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
    ],
)

cc_library(
    name = "stable_delegate_registry",
    hdrs = ["stable_delegate_registry.h"],
    visibility = experimental_acceleration_api_allowlist(),
    deps = [
        "//tensorflow/lite/core/acceleration/configuration:stable_delegate_registry",
    ],
)
