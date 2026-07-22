# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "online_helper_delegate",
    srcs = ["online_helper_delegate.cc"],
    hdrs = ["online_helper_delegate.h"],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "delegate_compatibility_checker_base",
    srcs = ["delegate_compatibility_checker_base.cc"],
    hdrs = [
        "delegate_compatibility_checker_base.h",
    ],
    deps = [
        ":delegate_compatibility_checker_util",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tensorflow/lite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "delegate_compatibility_checker_util",
    hdrs = [
        "delegate_compatibility_checker_util.h",
    ],
)
