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
    name = "litert_compiler_plugin",
    hdrs = ["litert_compiler_plugin.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
    ],
)

cc_library(
    name = "litert_compiler_plugin_api",
    hdrs = ["litert_compiler_plugin_api.h"],
    deps = [
        ":litert_compiler_plugin",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_dispatch_c_api",
    hdrs = [
        "litert_dispatch.h",
        "litert_dispatch_api.h",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_any",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_event",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/runtime/dispatch",
    ],
)

# This test verifies that the C API header files can build via C compiler.
cc_test(
    name = "litert_vendor_c_api_common_test",
    srcs = ["litert_vendor_c_api_common_test.c"],
    copts = ["--std=c11"],
    linkopts = ["-ldl"],
    deps = [
        ":litert_compiler_plugin",
        ":litert_compiler_plugin_api",
        ":litert_dispatch_c_api",
    ],
)

exports_files(srcs = glob(["litert_*.h"]))
