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

# Description:
# The cuDNN Frontend API is a C++ header-only library that demonstrates how
# to use the cuDNN C backend API.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE.txt"])

filegroup(
    name = "cudnn_frontend_header_files",
    srcs = glob([
        "include/**",
    ]),
)

cc_library(
    name = "cudnn_frontend",
    hdrs = [":cudnn_frontend_header_files"],
    include_prefix = "third_party/cudnn_frontend",
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
    ],
)
