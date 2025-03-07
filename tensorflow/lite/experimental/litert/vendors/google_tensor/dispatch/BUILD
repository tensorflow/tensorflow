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

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "copy_file", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

litert_dynamic_lib(
    name = "dispatch_api",
    srcs = [
        "dispatch_api.cc",
        "litert_dispatch_device_context.cc",
        "litert_dispatch_graph.cc",
        "litert_dispatch_invocation_context.cc",
        "southbound.cc",
    ],
    hdrs = [
        "dispatch_api.h",
        "litert_dispatch_device_context.h",
        "litert_dispatch_graph.h",
        "litert_dispatch_invocation_context.h",
        "litert_dispatch_metrics.h",
        "southbound.h",
        # copybara:uncomment "//third_party/odml/infra/southbound:sb_api.h",
    ],
    export_litert_only = True,
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    shared_lib_name = "dispatch_api_so",
    so_name = "libLiteRtDispatch_GoogleTensor.so",
    tags = [
        # Don't build/test in OSS until Southbound is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_any",
        "//tensorflow/lite/experimental/litert/c:litert_runtime_c_api_shared_lib",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/core/util:tensor_type_util",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# This is cc_library target for `libLiteRtDispatch_GoogleTensor.so`.
cc_library(
    name = "dispatch_api_shared_lib",
    srcs = [":dispatch_api_so"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

# Copies the shared library so that it is available for use in test data as libLiteRtDispatch_GoogleTensor.so.
copy_file(
    name = "copy_dispatch_api_so",
    src = "//tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch:dispatch_api_so",
    target = "libLiteRtDispatch_GoogleTensor.so",
)

cc_test(
    name = "dispatch_api_google_tensor_test",
    srcs = [
        "dispatch_api_google_tensor_test.cc",
    ],
    data = [
        ":dispatch_api_so",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        # Don't build/test in OSS until Southbound is available.
        "nobuilder",
        "no_oss",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_any",
        "//tensorflow/lite/experimental/litert/core:filesystem",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:simple_model_npu",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_dispatch_c_api",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dispatch_api_async_google_tensor_test",
    srcs = [
        "dispatch_api_async_google_tensor_test.cc",
    ],
    data = [
        ":dispatch_api_so",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        # Don't build/test in OSS until Southbound is available.
        "nobuilder",
        "no_oss",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/darwinn/driver_shared/fence:fence_test_util",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_event",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_any",
        "//tensorflow/lite/experimental/litert/core:filesystem",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:simple_model_npu",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_dispatch_c_api",
    ],
)
