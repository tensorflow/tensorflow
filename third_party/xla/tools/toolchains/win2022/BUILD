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

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

java_runtime(
    name = "windows_jdk8",
    srcs = [],
    java_home = "C:/openjdk",
)

# Register a Windows 2022 (Clang) platform.
platform(
    name = "windows_ltsc2022_clang",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/tf-win2022@sha256:915cb093630432c38b028f56bd31116a5559ebbc688d427b6092d86828ae03bc",
        "OSFamily": "Windows",
        "Pool": "win2022",
        "dockerNetwork": "off",
        "cache-silo-key": "20251105-1762360217",
    },
)

# TODO(b/457398048): Deprecate the above platform once this one is working.
platform(
    name = "windows_ltsc2022_clang_new",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/tf-win2022@sha256:915cb093630432c38b028f56bd31116a5559ebbc688d427b6092d86828ae03bc",
        "OSFamily": "Windows",
        "Pool": "windows-2022-pool",
        "dockerNetwork": "off",
        "cache-silo-key": "20251105-1762360217",
    },
)

# RBE platform for ML Actions.
platform(
    name = "windows_ltsc2022_clang_ml_actions",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
    exec_properties = {
        "container-image": "docker://us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build-windows@sha256:023fa901fda70be8890b05689d0a667287692c1f59d0abe4add7208ec5a12707",
        "OSFamily": "Windows",
        "Pool": "windows-2022-pool",
        "dockerNetwork": "off",
        "cache-silo-key": "20260115",
    },
)
