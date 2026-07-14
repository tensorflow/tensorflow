# Copyright 2024 The OpenXLA Authors.
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
"""Test utils for python tests in XLA."""
import os
import pathlib


def xla_src_root() -> pathlib.Path:
  """Gets the path to the root of the XLA source tree."""
  is_oss = "BAZEL_TEST" in os.environ
  test_srcdir = os.environ["TEST_SRCDIR"]
  test_workspace = os.environ["TEST_WORKSPACE"]
  if is_oss:
    return pathlib.Path(test_srcdir) / test_workspace
  else:
    return pathlib.Path(test_srcdir) / test_workspace / "third_party" / "xla"
