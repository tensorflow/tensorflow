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

"""Helper function to get canonical repo names in Bzlmod."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def get_canonical_repo_name(apparent_repo_name):
    """Returns the canonical repo name for the given apparent repo name."""

    # Internally, Label("//:foo") stringifies to "//:foo".
    # In Bazel with Bzlmod enabled, it stringifies to "@@//:foo" or similar.
    if not str(Label("//:foo")).startswith("@@"):
        # Internally, we don't need to canonicalize repo names.
        return apparent_repo_name

    return Label("@" + apparent_repo_name).repo_name
