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

"""Common libraries for IFRT proxy."""

# This file is used in OSS only. It is not transformed by copybara. Therefore all paths in this
# file are OSS paths.

load("@rules_cc//cc:cc_library.bzl", _cc_library = "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

# IMPORTANT: Do not remove this load statement. We rely on that //xla/tsl doesn't exist in g3
# to prevent g3 .bzl files from loading this file.
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def ifrt_proxy_cc_test(
        **kwargs):
    xla_cc_test(
        **kwargs
    )

default_ifrt_proxy_visibility = ["//visibility:public"]

ifrt_proxy_grpc_client_visibility = default_ifrt_proxy_visibility

def cc_library(**attrs):
    _cc_library(**attrs)
