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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

exports_files(["LICENSE"])

license(
    name = "license",
    package_name = "vim",
    license_kinds = [
        "@rules_license//licenses/spdx:Vim",
    ],
    license_text = "LICENSE",
    package_url = "https://github.com/vim/vim",
)

cc_binary(
    name = "xxd",
    srcs = [
        "xxd.c",
    ],
    local_defines = select({
        "@platforms//os:windows": ["WIN32=1"],
        "//conditions:default": [],
    }),
)
