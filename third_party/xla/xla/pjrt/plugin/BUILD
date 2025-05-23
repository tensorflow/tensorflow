# Copyright 2017 The OpenXLA Authors.
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

# Configuration file for an XLA plugin.
#
# please don't check in changes to this file. to prevent changes appearing
# in git status, use:
#
# git update-index --assume-unchanged tensorflow/compiler/xla/pjrt/plugin/BUILD
#
# To add additional devices to the XLA subsystem, add targets to the
# dependency list in the 'plugin' target. For instance:
#
#   deps = ["//xla/pjrt/plugin/example:plugin_lib"],
#
# ** Please don't remove this file - it is supporting some 3rd party plugins **

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "plugin",
    deps = [
        #"//xla/pjrt/plugin/example:example_lib",
    ],
)

cc_library(
    name = "plugin_names",
    hdrs = ["plugin_names.h"],
    deps = ["@com_google_absl//absl/strings:string_view"],
)

cc_library(
    name = "static_registration",
    hdrs = ["static_registration.h"],
)
