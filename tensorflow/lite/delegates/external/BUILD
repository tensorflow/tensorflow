# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "external_delegate_interface",
    hdrs = ["external_delegate_interface.h"],
    deps = [
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "external_delegate",
    srcs = ["external_delegate.cc"],
    hdrs = ["external_delegate.h"],
    deps = [
        ":external_delegate_interface",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:shared_library",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
    ],
)

exports_files([
    "external_delegate.h",
])
