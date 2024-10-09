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

load("//tensorflow/lite/experimental/lrt/qnn_sdk:ungrte_qnn.bzl", "ungrte_cc_library_with_qnn", "ungrte_cc_test_with_qnn")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/lrt:__subpackages__"],
)

ungrte_cc_library_with_qnn(
    name = "log",
    srcs = ["log.cc"],
    hdrs = ["log.h"],
    deps = [
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
    ],
)

ungrte_cc_library_with_qnn(
    name = "qnn_manager",
    srcs = ["qnn_manager.cc"],
    hdrs = ["qnn_manager.h"],
    deps = [
        ":log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
        "//tensorflow/lite/experimental/lrt/core:api_internal",
        "//tensorflow/lite/experimental/lrt/core:dynamic_loading",
        "//tensorflow/lite/experimental/lrt/core:logging",
    ],
)

ungrte_cc_test_with_qnn(
    name = "load_qnn_test",
    srcs = ["load_qnn_test.cc"],
    # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
    tags = ["nosan"],
    deps = [
        ":qnn_manager",
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite/experimental/lrt/test:common",
    ],
)
