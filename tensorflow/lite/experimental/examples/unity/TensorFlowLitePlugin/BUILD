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

load("//devtools/unity3d/unity_build:unity_build_project.bzl", "unity_build_project")
load("//tools/build_defs/build_test:build_test.bzl", "build_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:LICENSE"])

licenses(["notice"])

unity_build_project(
    name = "build_check",
    additional_deps = [
        "//third_party/mono:System.Core",
    ],
    allow_unsafe_blocks = True,
    tags = [
        "notsan",  # tsan is incompatible with certain Unity builds.
    ],
)

# TODO(b/136186242): Add unity_style_check after resolving style issues.
build_test(
    name = "tap_test",
    tags = [
        "notsan",  # tsan is incompatible with certain Unity builds.
    ],
    targets = [
        ":build_check",
    ],
)
