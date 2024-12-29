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

cc_test(
    name = "jit_compilation_qualcomm_test",
    srcs = ["jit_compilation_qualcomm_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_dispatch_delegate",
        "//tensorflow/lite/experimental/litert/cc:litert_compiled_model",
        "//tensorflow/lite/experimental/litert/cc:litert_environment",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/compiler/plugin:compiler_plugin",
        "//tensorflow/lite/experimental/litert/runtime:external_litert_buffer_context",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:simple_model_npu",
        "//tensorflow/lite/experimental/litert/test:test_macros",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "jit_compilation_mediatek_test",
    srcs = ["jit_compilation_mediatek_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler:compiler_plugin_so",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/dispatch:dispatch_api_so",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_dispatch_delegate",
        "//tensorflow/lite/experimental/litert/cc:litert_compiled_model",
        "//tensorflow/lite/experimental/litert/cc:litert_environment",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/compiler/plugin:compiler_plugin",
        "//tensorflow/lite/experimental/litert/runtime:external_litert_buffer_context",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:simple_model_npu",
        "//tensorflow/lite/experimental/litert/test:test_macros",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
