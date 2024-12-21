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
    name = "schema",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "byte_code_asset",
    srcs = ["byte_code_asset.cc"],
    hdrs = ["byte_code_asset.h"],
    deps = [
        ":schema",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/core/model",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "schema_test",
    srcs = ["schema_test.cc"],
    deps = [
        ":schema",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "byte_code_asset_test",
    srcs = ["byte_code_asset_test.cc"],
    deps = [
        ":byte_code_asset",
        ":schema",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/test:test_macros",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
