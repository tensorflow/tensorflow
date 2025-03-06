# Copyright 2025 Google LLC.
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
    default_visibility = [
        "//tensorflow/lite/experimental/litert:__subpackages__",
    ],
)

cc_library(
    name = "auto_registration",
    srcs = ["auto_registration.cc"],
    hdrs = ["auto_registration.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_shared_library",
        "//tensorflow/lite/experimental/litert/core:environment",
        "//tensorflow/lite/experimental/litert/runtime/accelerators/dispatch:dispatch_accelerator",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "accelerator_implementation_helper",
    hdrs = ["accelerator_implementation_helper.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_accelerator",
        "//tensorflow/lite/experimental/litert/c:litert_accelerator_compilation_options",
        "//tensorflow/lite/experimental/litert/c:litert_accelerator_registration",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/runtime:accelerator_model_compilation_data",
        "@com_google_absl//absl/strings:string_view",
    ],
)
