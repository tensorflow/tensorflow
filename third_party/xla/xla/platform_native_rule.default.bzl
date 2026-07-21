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

"""Default platform native rule wrapper (passthrough for OSS)."""

load("//xla:native_test.bzl", "native_test")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def platform_native_rule(
        name,  # @unused
        backend):  # @unused
    """Factory function returning a lambda that simply calls native_test.

    Args:
        name: string. Ignored.
        backend: string. Ignored.

    Returns:
        A lambda function that takes `name` and `**kwargs` and calls `native_test`.
    """
    return lambda name, **kwargs: native_test(name = name, **kwargs)
