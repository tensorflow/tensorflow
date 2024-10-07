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
    default_visibility = ["//tensorflow/lite/experimental/lrt:__subpackages__"],
)

cc_library(
    name = "common",
    testonly = 1,
    srcs = [
        "common.cc",
    ],
    hdrs = [
        "common.h",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "simple_model",
    testonly = 1,
    srcs = [],
    hdrs = [
        "testdata/simple_model_test_vectors.h",
    ],
    data = [
        "testdata/simple_model.tflite",
        "testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
    ],
)
