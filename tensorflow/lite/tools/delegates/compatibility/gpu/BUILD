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
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_test(
    name = "gpu_delegate_compatibility_checker_test",
    srcs = [
        "gpu_delegate_compatibility_checker_test.cc",
    ],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/conv3d_huge_im2col.bin",
    ],
    tags = ["no_oss"],  #TODO(b/276295784): Re-enable when fixed.
    deps = [
        ":gpu_delegate_compatibility_checker",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_delegate_compatibility_checker",
    srcs = ["gpu_delegate_compatibility_checker.cc"],
    hdrs = [
        "gpu_delegate_compatibility_checker.h",
    ],
    deps = [
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/tools/delegates/compatibility/common:delegate_compatibility_checker_base",
        "//tensorflow/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//tensorflow/lite/tools/versioning:gpu_compatibility",
        "//tensorflow/lite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
    ],
)
