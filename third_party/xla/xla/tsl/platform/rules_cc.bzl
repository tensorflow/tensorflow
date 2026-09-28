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

"""Provides an indirection layer to bazel cc_rules"""

load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_TSL_PLATFORM_RULES_CC_USERS",
)
load(
    "//xla/tsl/platform/default:rules_cc.bzl",
    _cc_binary = "cc_binary",
    _cc_import = "cc_import",
    _cc_library = "cc_library",
    _cc_shared_library = "cc_shared_library",
    _cc_test = "cc_test",
    _default_compatible_with = "default_compatible_with",
)

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_TSL_PLATFORM_RULES_CC_USERS)

cc_binary = _cc_binary
cc_import = _cc_import
cc_library = _cc_library
cc_shared_library = _cc_shared_library
cc_test = _cc_test
default_compatible_with = _default_compatible_with
