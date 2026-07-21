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

"""If we're building a nightly, we use this to pass a timestamp for the wheel version."""

def _nightly_timestamp_impl(rctx):
    timestamp_val = rctx.getenv("XLA_NIGHTLY_TIMESTAMP", "")  # Default to ""

    # Smuggle the value via a new .bzl file
    if timestamp_val:
        rctx.file(
            "timestamp.bzl",
            content = 'XLA_NIGHTLY_TIMESTAMP = ".dev{}"'.format(timestamp_val),
        )
    else:
        rctx.file(
            "timestamp.bzl",
            content = 'XLA_NIGHTLY_TIMESTAMP = ""',
        )

    # Create a BUILD file to make timestamp.bzl addressable
    rctx.file("BUILD.bazel", content = "")

nightly_timestamp_repo = repository_rule(
    implementation = _nightly_timestamp_impl,
    environ = ["XLA_NIGHTLY_TIMESTAMP"],
)

# bzlmod implementation
def _nightly_timestamp_ext_impl(mctx):  # @unused
    nightly_timestamp_repo(
        name = "nightly_timestamp",
    )

nightly_timestamp_repo_bzlmod = module_extension(
    implementation = _nightly_timestamp_ext_impl,
    environ = ["XLA_NIGHTLY_TIMESTAMP"],
)
