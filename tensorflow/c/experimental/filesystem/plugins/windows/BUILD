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

# Experimental windows filesystem plugin.
load("//tensorflow:tensorflow.bzl", "get_win_copts", "tf_cc_shared_object")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# Filesystem implementation for Windows environment
tf_cc_shared_object(
    name = "windows_filesystem.dll",
    framework_so = [],
    linkstatic = False,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    visibility = ["//visibility:public"],
    deps = [":windows_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "windows_filesystem_impl",
    srcs = ["windows_filesystem.cc"],
    copts = get_win_copts(),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = ["//tensorflow/c/experimental/filesystem:filesystem_interface"],
)
