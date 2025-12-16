# Copyright 2025 The OpenXLA Authors.
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
# ============================================================================

load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

string_flag(
    name = "sanitizer",
    build_setting_default = "none",
    values = [
        "none",
        "asan",
        "tsan",
    ],
)

config_setting(
    name = "asan",
    flag_values = {"//build_tools/rocm:sanitizer": "asan"},
)

config_setting(
    name = "tsan",
    flag_values = {"//build_tools/rocm:sanitizer": "tsan"},
)

filegroup(
    name = "sanitizer_ignore_lists",
    srcs = select({
        ":asan": [
            "asan_ignore_list.txt",
            "lsan_ignore_list.txt",
        ],
        ":tsan": ["tsan_ignore_list.txt"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

genrule(
    name = "san_wrapper_script",
    srcs = [":sanitizer_ignore_lists"],
    outs = ["san_wrapper.sh"],
    cmd = """
      echo '#!/bin/bash' > $@
      echo 'exec "$$@"' >> $@
      chmod +x $@
    """,
)

# this wrapper ensures the test target
# take into account any changes in the ignore list files
sh_binary(
    name = "sanitizer_wrapper",
    srcs = [":san_wrapper_script"],
    data = [":sanitizer_ignore_lists"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "parallel_gpu_execute",
    srcs = ["parallel_gpu_execute.sh"],
    data = [
        ":sanitizer_ignore_lists",
        "@local_config_rocm//rocm:rocminfo",
    ],
    visibility = ["//visibility:public"],
)
