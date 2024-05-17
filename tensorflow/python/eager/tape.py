# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Gradient tape utilities."""

from tensorflow.python import pywrap_tfe


class Tape(object):
  """Represents a gradient propagation trace."""

  __slots__ = ["_tape"]

  def __init__(self, tape):
    self._tape = tape

  def watched_variables(self):
    return pywrap_tfe.TFE_Py_TapeWatchedVariables(self._tape)


def push_new_tape(persistent=False, watch_accessed_variables=True):
  """Pushes a new tape onto the tape stack."""
  tape = pywrap_tfe.TFE_Py_TapeSetNew(persistent, watch_accessed_variables)
  return Tape(tape)


def push_tape(tape):
  """Pushes an existing tape onto the tape stack."""
  pywrap_tfe.TFE_Py_TapeSetAdd(tape._tape)  # pylint: disable=protected-access


def watch(tape, tensor):
  """Marks this tensor to be watched by the given tape."""
  pywrap_tfe.TFE_Py_TapeWatch(tape._tape, tensor)  # pylint: disable=protected-access


def default_get_variables(variable):
  return [variable]

# Gets a list of changed variables. Can be overriden using
# register_variables_override. An example of overriding is for getting the
# varibles within a distributed context.
_variables_override = default_get_variables


def register_watched_variable_resolver(resolver):
  """Registers the resolver to be used to get the list of variables to watch.

  Args:
    resolver: callable, takes a Variable and returns a list of Variables that
      shall be watched.
  """
  global _variables_override
  assert _variables_override is default_get_variables
  _variables_override = resolver


def watch_variable(tape, variable):
  """Marks this variable to be watched by the given tape."""
  variables = _variables_override(variable)
  for var in variables:
    pywrap_tfe.TFE_Py_TapeWatchVariable(tape._tape, var)  # pylint: disable=protected-access
    pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)


def variable_accessed(variable):
  """Notifies all tapes in the stack that a variable has been accessed.

  Args:
    variable: variable to be watched.
  """
  variables = _variables_override(variable)
  for var in variables:
    pywrap_tfe.TFE_Py_TapeVariableAccessed(var)
    pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)


def variables_accessed(variables):
  """Notifies all tapes in the stack that variables have been accessed.

  Only trainable variables are marked as accessed.

  Args:
    variables: iterable of variables to mark as accessed.
  """
  accessed = []
  for variable in variables:
    if variable.trainable:
      accessed.extend(_variables_override(variable))

  for var in accessed:
    pywrap_tfe.TFE_Py_TapeVariableAccessed(var)
    pywrap_tfe.TFE_Py_VariableWatcherVariableAccessed(var)


def pop_tape(tape):
  """Pops the given tape in the stack."""
  pywrap_tfe.TFE_Py_TapeSetRemove(tape._tape)  # pylint: disable=protected-access
