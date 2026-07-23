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

"""Description: BUILD file for shell script to install bazel using the installer.
"""

load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    srcs = glob(["*.sh"]),
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "install_bazel",
    srcs = ["install_bazel.sh"],
)

sh_binary(
    name = "install_bazel_from_source",
    srcs = ["install_bazel_from_source.sh"],
)
