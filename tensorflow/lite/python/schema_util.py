# Lint as: python2, python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Schema utilities to get builtin code from operator code."""

from tensorflow.python.util import all_util


def get_builtin_code_from_operator_code(opcode):
  """Return the builtin code of the given operator code.

  The following method is introduced to resolve op builtin code shortage
  problem. The new builtin operator will be assigned to the extended builtin
  code field in the flatbuffer schema. Those methods helps to hide builtin code
  details.

  Args:
    opcode: Operator code.

  Returns:
    The builtin code of the given operator code.
  """
  # Access BuiltinCode() method first if available.
  if hasattr(opcode, 'BuiltinCode') and callable(opcode.BuiltinCode):
    return max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())

  return max(opcode.builtinCode, opcode.deprecatedBuiltinCode)


_allowed_symbols = [
    'get_builtin_code_from_operator_code',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
