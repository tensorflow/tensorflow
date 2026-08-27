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

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "stream_ops_op_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/tfrt/kernels:stream_ops",
        "//tensorflow/core/tfrt/ops:stream_ops_op_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ifrt_program_ops_op_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/tfrt/kernels:ifrt_program_ops",
        "//tensorflow/core/tfrt/ops:ifrt_program_ops_op_lib",
    ],
    alwayslink = 1,
)
