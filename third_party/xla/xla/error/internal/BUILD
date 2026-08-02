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

load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":internal_users",
    ],
    licenses = ["notice"],
)

package_group(
    name = "internal_users",
    packages = [
        "//xla/error/...",
    ],
)

cc_library(
    name = "check_impl",
    hdrs = ["check_impl.h"],
    deps = [
        ":check_helper",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "check_helper",
    hdrs = ["check_helper.h"],
    deps = [
        "//xla/error:debug_me_context_util",
        "//xla/error:error_codes",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)
