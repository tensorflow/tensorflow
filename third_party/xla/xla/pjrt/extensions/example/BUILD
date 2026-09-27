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
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

# Only the plugin code should be accessing the opaque types, do not make these visible to the
# consumer of the plugin.
package_group(
    name = "extension_private",
    packages = [
        "//xla/pjrt/plugin/...",
    ],
)

cc_library(
    name = "example_extension_cpp",
    hdrs = ["example_extension_cpp.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "example_extension",
    srcs = ["example_extension.cc"],
    hdrs = ["example_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":example_extension_private",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "example_extension_private",
    hdrs = ["example_extension_private.h"],
    visibility = [":extension_private"],
    deps = [":example_extension_cpp"],
)
