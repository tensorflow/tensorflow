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

# Tests for the C++ header-only SavedModelAPI.
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_test(
    name = "saved_model_api_test",
    srcs = [
        "saved_model_api_test.cc",
    ],
    data = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/cc/experimental/base/public:runtime",
        "//tensorflow/cc/experimental/base/public:runtime_builder",
        "//tensorflow/cc/experimental/base/public:status",
        "//tensorflow/cc/saved_model/experimental/public:saved_model_api",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings:string_view",
    ],
)
