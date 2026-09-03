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

# Test helper binaries to use in tests.
#   These are provided to avoid linux-only binaries in tests.
#   Thus helping write cross platform tests.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla/tsl/platform:__pkg__",
    ],
    licenses = ["notice"],
)

cc_binary(
    name = "test_echo",
    srcs = ["test_echo.cc"],
)

cc_binary(
    name = "test_noop",
    srcs = ["test_noop.cc"],
)

cc_binary(
    name = "test_echo_argv_1",
    srcs = ["test_echo_argv_1.cc"],
)

cc_binary(
    name = "test_stderr",
    srcs = ["test_stderr.cc"],
)

cc_binary(
    name = "test_crash",
    srcs = ["test_crash.cc"],
)

cc_binary(
    name = "test_pwd",
    srcs = ["test_pwd.cc"],
)

cc_binary(
    name = "test_getenv",
    srcs = ["test_getenv.cc"],
)
