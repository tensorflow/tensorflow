# Copyright 2026 The OpenXLA Authors.
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

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

genrule(
    name = "viewer_server_h",
    srcs = ["viewer_server.py"],
    outs = ["viewer_server.h"],
    cmd = "echo 'const char* kPythonServerCode = R\"python(' > $@ && cat $< >> $@ && echo ')python\";' >> $@",
    compatible_with = get_compatible_with_portable(),
)

genrule(
    name = "viewer_html_h",
    srcs = ["viewer.html"],
    outs = ["viewer_html.h"],
    cmd = "echo 'inline const char* kViewerHtmlCode = R\"html(' > $@ && cat $< >> $@ && echo ')html\";' >> $@",
    compatible_with = get_compatible_with_portable(),
)

genrule(
    name = "simple_viewer_html_h",
    srcs = ["simple_viewer.html"],
    outs = ["simple_viewer_html.h"],
    cmd = "echo 'inline const char* kSimpleViewerHtmlCode = R\"html(' > $@ && cat $< >> $@ && echo ')html\";' >> $@",
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "fusion_explorer_embed",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = [
        "simple_viewer_html.h",
        "viewer_html.h",
        "viewer_server.h",
    ],
)
