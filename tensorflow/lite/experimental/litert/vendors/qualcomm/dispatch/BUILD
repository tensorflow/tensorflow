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
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_invocation_context.cc",
    ],
    hdrs = [
        "litert_dispatch_device_context.h",
        "litert_dispatch_invocation_context.h",
        "registry.h",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/core:utils",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_dispatch_c_api",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:context_binary_info",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
    ],
)

cc_shared_library(
    name = "dispatch_api_shared",
    shared_lib_name = "libLiteRtDispatch.so",
    visibility = ["//visibility:public"],
    deps = [":dispatch_api"],
)

cc_test(
    name = "dispatch_api_qualcomm_test",
    srcs = [
        "dispatch_api_qualcomm_test.cc",
    ],
    data = [
        ":dispatch_api_shared",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    tags = [
        "no-remote-exec",
        "no_oss",
        "notap",
    ],
    deps = [
        ":dispatch_api",
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
