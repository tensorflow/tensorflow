# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test for Python APIs accessing MHLO types."""

# pylint: disable=wildcard-import,undefined-variable

from mlir.dialects.mhlo import *
from mlir.ir import *


def run(f):
  with Context() as context:
    register_mhlo_dialect(context)
    f()
  return f


@run
def test_token_type():
  """Check that the Token type is available."""
  assert str(TokenType.get()) == "!mhlo.token"
