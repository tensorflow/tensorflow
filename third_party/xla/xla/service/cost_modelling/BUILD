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

# Description:
#   A common directory for cost modelling related code.

load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "op_cost",
    hdrs = ["op_cost.h"],
    tags = if_google(["avoid_dep"]),
    deps = [
        "//xla/hlo/analysis:op_cost",
    ],
)

cc_library(
    name = "op_cost_test_utils",
    testonly = True,
    hdrs = ["op_cost_test_utils.h"],
    tags = if_google(["avoid_dep"]),
    deps = [
        "//xla/hlo/analysis:op_cost_test_utils",
    ],
)
