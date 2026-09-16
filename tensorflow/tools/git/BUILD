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
# Contains script to generate tensorflow/core/util/version_info.cc
# Also contains information about git repository deposited by configure
# in gen/...

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("//tensorflow:tensorflow.bzl", "if_oss")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

exports_files(["gen_git_source.py"])

py_binary(
    name = "gen_git_source",
    srcs = ["gen_git_source.py"],
    strict_deps = True,
    tags = if_oss(["no-remote-exec"]),
)
