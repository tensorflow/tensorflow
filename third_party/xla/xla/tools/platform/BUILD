# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(
        ["//xla:__subpackages__"],
    ),
    licenses = ["notice"],
)

exports_files(
    ["xprof_loader.h"],
    visibility = internal_visibility(
        ["//xla/tools/platform:__subpackages__"],
    ),
)

cc_library(
    name = "xprof_loader",
    hdrs = ["xprof_loader.h"],
    compatible_with = [],
    textual_hdrs = ["xprof_loader.h"],
    deps = [
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status:statusor",
    ] + if_google(
        ["//xla/tools/platform/google:xprof_loader"],
        ["//xla/tools/platform/default:xprof_loader"],
    ),
)
