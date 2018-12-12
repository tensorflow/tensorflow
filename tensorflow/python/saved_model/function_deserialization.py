# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tools for deserializing PolymorphicFunctions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function


def recreate_polymorphic_function(
    saved_polymorphic_function, defined_functions):
  """Creates a PolymorphicFunction which runs restored function definitions."""
  @def_function.function
  def restored_function(*args):
    """Calls a restored function."""
    # Try calling each function, return a value from the first one whose
    # signature matches.
    # TODO(allenl): Consider re-populating the function cache directly.
    # TODO(allenl): Functions saved with input_signatures should revive with
    # input_signatures.
    for monomorphic_function in saved_polymorphic_function.monomorphic_function:
      try:
        # TODO(allenl): Passing an explicit name here prevents invalid name
        # errors. We should replace this with something based on the actual
        # Python function name.
        return defined_functions[monomorphic_function.concrete_function](
            *args, name="imported_function")
      except ValueError:
        continue
    raise AssertionError(
        "Could not find matching function to call for arguments: %s" % (args,))
  return restored_function
