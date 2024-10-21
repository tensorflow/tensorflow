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
    name = "dispatch",
    srcs = [
        "litert_dispatch.cc",
    ],
    hdrs = [
    ],
    deps = [
        "//tensorflow/lite/experimental/lrt/c:litert_c_api",
        "//tensorflow/lite/experimental/lrt/c:litert_tensor_buffer_c_api",
        "//tensorflow/lite/experimental/lrt/core:logging",
        "//tensorflow/lite/experimental/lrt/core:tensor_buffer",
        "//tensorflow/lite/experimental/lrt/core:utils",
        "//tensorflow/lite/experimental/lrt/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dispatch_delegate",
    srcs = [
        "dispatch_delegate.cc",
        "dispatch_delegate_kernel.cc",
    ],
    hdrs = [
        "dispatch_delegate_kernel.h",
        "dispatch_delegate_options.h",
    ],
    deps = [
        ":dispatch",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/experimental/lrt/c:litert_c_api",
        "//tensorflow/lite/experimental/lrt/c:litert_dispatch_delegate",
        "//tensorflow/lite/experimental/lrt/c:litert_tensor_buffer_c_api",
        "//tensorflow/lite/experimental/lrt/cc:litert_cc_api",
        "//tensorflow/lite/experimental/lrt/cc:litert_tensor_buffer_cc_api",
        "//tensorflow/lite/experimental/lrt/core:logging",
        "//tensorflow/lite/experimental/lrt/core:tensor_buffer",
        "//tensorflow/lite/experimental/lrt/core:tfl_utils",
        "//tensorflow/lite/experimental/lrt/core:utils",
        "//tensorflow/lite/experimental/lrt/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "dispatch_delegate_google_tensor_test",
    srcs = ["dispatch_delegate_google_tensor_test.cc"],
    data = [
        "//tensorflow/lite/experimental/lrt/vendors/google_tensor/dispatch:dispatch_api_shared",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":dispatch",
        ":dispatch_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/lrt/c:litert_c_api",
        "//tensorflow/lite/experimental/lrt/c:litert_dispatch_delegate",
        "//tensorflow/lite/experimental/lrt/test:common",
        "//tensorflow/lite/experimental/lrt/test:simple_model_npu",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform",
    ],
)

cc_test(
    name = "dispatch_delegate_qualcomm_test",
    srcs = ["dispatch_delegate_qualcomm_test.cc"],
    data = [
        "//tensorflow/lite/experimental/lrt/vendors/qualcomm/dispatch:dispatch_api_shared",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":dispatch",
        ":dispatch_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/lrt/c:litert_c_api",
        "//tensorflow/lite/experimental/lrt/c:litert_dispatch_delegate",
        "//tensorflow/lite/experimental/lrt/test:common",
        "//tensorflow/lite/experimental/lrt/test:simple_model_npu",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform",
    ],
)
