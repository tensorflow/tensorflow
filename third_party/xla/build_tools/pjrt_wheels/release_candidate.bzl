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

"""If we're building a release candidate, we use this to pass a rc number for the wheel version."""

def _rc_number_impl(rctx):
    rc_number = rctx.getenv("XLA_RC_NUMBER", "")  # Default to ""

    # Smuggle the value via a new .bzl file
    if rc_number:
        rctx.file(
            "rc_number.bzl",
            content = 'XLA_RC_NUMBER = "{}"'.format(rc_number),
        )
    else:
        rctx.file(
            "rc_number.bzl",
            content = 'XLA_RC_NUMBER = ""',
        )

    # Create a BUILD file to make timestamp.bzl addressable
    rctx.file("BUILD.bazel", content = "")

rc_number_repo = repository_rule(
    implementation = _rc_number_impl,
    environ = ["XLA_RC_NUMBER"],
)

# bzlmod implementation
def _rc_number_ext_impl(mctx):  # @unused
    rc_number_repo(
        name = "rc_number",
    )

rc_number_repo_bzlmod = module_extension(
    implementation = _rc_number_ext_impl,
    environ = ["XLA_RC_NUMBER"],
)
