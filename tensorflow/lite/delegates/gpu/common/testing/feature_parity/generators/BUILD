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
    default_testonly = 1,
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "all_generators",
    hdrs = [
        "add.h",
    ],
    deps = [
        ":add",
        "//tensorflow/lite/delegates/gpu/common/testing/feature_parity:utils",
    ],
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/delegates/gpu/common/testing/feature_parity:utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)
