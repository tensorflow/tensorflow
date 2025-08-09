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

load("//xla:xla.default.bzl", "xla_cc_test")
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
    srcs = ["static_registration.cc"],
    hdrs = ["static_registration.h"],
    deps = [
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "static_registration_test",
    srcs = ["static_registration_test.cc"],
    deps = [
        ":static_registration",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dynamic_registration",
    srcs = ["dynamic_registration.cc"],
    hdrs = ["dynamic_registration.h"],
    deps = [
        "//xla/pjrt:pjrt_api",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "dynamic_registration_test",
    srcs = ["dynamic_registration_test.cc"],
    # We must link this statically to prevent issues with symbol resolution between the loaded .so
    # for the test and the .so file.
    linkstatic = True,
    deps = [
        ":dynamic_registration",
        ":dynamic_registration_test_helper",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "dynamic_registration_test_helper",
    testonly = True,
    srcs = ["dynamic_registration_test_helper.cc"],
    data = [
        "//xla/pjrt/plugin/example_plugin:pjrt_c_api_myplugin_plugin.so",
    ],
    deps = [
        ":dynamic_registration",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:test",
        "@local_tsl//tsl/platform:path",
    ],
    alwayslink = 1,
)
