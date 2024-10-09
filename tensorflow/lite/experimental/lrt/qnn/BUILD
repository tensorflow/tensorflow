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
    default_visibility = ["//visibility:private"],
)

ungrte_cc_library_with_qnn(
    name = "qnn_compiler_plugin",
    srcs = ["qnn_compiler_plugin.cc"],
    visibility = ["//tensorflow/lite/experimental/lrt:__subpackages__"],
    deps = [
        ":qnn_compose_graph",
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
        "//tensorflow/lite/experimental/lrt/core:api_internal",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/qnn_sdk:qnn_manager",
    ],
)

ungrte_cc_test_with_qnn(
    name = "qnn_compiler_plugin_test",
    srcs = [
        "qnn_compiler_plugin_test.cc",
    ],
    # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
    tags = ["nosan"],
    deps = [
        ":qnn_compiler_plugin",  # buildcleaner: keep
        "@com_google_absl//absl/log:absl_check",
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
        "//tensorflow/lite/experimental/lrt/core:api_internal",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/qnn_sdk:qnn_manager",
        "//tensorflow/lite/experimental/lrt/test:common",
    ],
)

ungrte_cc_library_with_qnn(
    name = "qnn_compose_graph",
    srcs = ["qnn_compose_graph.cc"],
    hdrs = ["qnn_compose_graph.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/qnn/IR:qnn_op",
        "//tensorflow/lite/experimental/lrt/qnn/IR:qnn_tensor",
        "//tensorflow/lite/experimental/lrt/qnn_sdk:qnn_manager",
    ],
)
