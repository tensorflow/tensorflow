# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implement the "specs" DSL for describing deep networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import importlib
import operator
import re

from six import exec_

QUOTED = re.compile(r"""
"([^"\\]|\\.)*" |
'([^'\\]|\\.)*'
""", re.VERBOSE)
KEYWORDS = re.compile(r"""\b(import|while|def|exec)\b""")


debug_ = False


def check_keywords(spec):
  """Check for common Python keywords in spec.

  This function discourages the use of complex constructs
  in TensorFlow specs; it doesn't completely prohibit them
  (if necessary, we could check the AST).

  Args:
      spec: spec string

  Raises:
      ValueError: raised if spec contains a prohibited keyword.
  """
  spec = re.sub(QUOTED, "", spec)
  match = re.search(KEYWORDS, spec)
  if match:
    raise ValueError("keyword '%s' found in spec" % match.group(1))


def get_positional(args, kw, kw_overrides=False):
  """Interpolates keyword arguments into argument lists.

  If `kw` contains keywords of the form "_0", "_1", etc., these
  are positionally interpolated into the argument list.

  Args:
      args: argument list
      kw: keyword dictionary
      kw_overrides: key/value pairs that override kw

  Returns:
      (new_args, new_kw), new argument lists and keyword dictionaries
      with values interpolated.
  """
  new_kw = {k: v for k, v in kw.items() if k[0] != "_"}
  if len(new_kw) == len(kw):
    return args, kw
  new_args = list(args)
  for key, value in kw.items():
    if key[0] != "_": continue
    index = int(key[1:])
    while len(new_args) <= index:
      new_args += [None]
    if kw_overrides or new_args[index] is None:
      new_args[index] = value
  return new_args, new_kw


class Composable(object):
  """A composable function.

  This defines the operators common to all composable objects.
  Currently defines copmosition (via "|") and repeated application
  (via "**"), and maps addition ("+") and multiplication ("*")
  as "(f + g)(x) = f(x) + g(x)".
  """

  def __or__(self, f):
    return Composition(self, f)

  def __add__(self, g):
    return Operator(operator.add, self, g)

  def __mul__(self, g):
    return Operator(operator.mul, self, g)

  def __pow__(self, n):
    assert n >= 0
    if n == 0:
      return Function(lambda x, *args, **kw: x)
    result = self
    for _ in range(n-1):
      result = Composition(result, self)
    return result


class Callable(Composable):
  """A composable function that simply defers to a callable function.
  """

  def __init__(self, f):
    self.f = f

  def funcall(self, x):
    return self.f(x)


class Operator(Composable):
  """A wrapper for an operator.

  This takes an operator and an argument list and returns
  the result of applying the operator to the results of applying
  the functions in the argument list.
  """

  def __init__(self, op, *args):
    self.op = op
    self.funs = args

  def funcall(self, x):
    outputs = [f.funcall(x) for f in self.funs]
    return self.op(*outputs)


class Function(Composable):
  """A composable function wrapper for a regular Python function.

  This overloads the regular __call__ operator for currying, i.e.,
  arguments passed to __call__ are remembered for the eventual
  function application.

  The final function application happens via the `of` method.
  """

  def __init__(self, f, *args, **kw):
    if not callable(f):
      raise ValueError("%s: is not callable" % f)
    self.f = f
    self.args = list(args)
    self.kw = kw

  def __call__(self, *args, **kw):
    new_args = list(args) + self.args
    new_kw = self.kw.copy()
    new_kw.update(kw)
    return Function(self.f, *new_args, **new_kw)

  # TODO(tmb) The `of` method may be renamed to `function`.
  def funcall(self, x):
    args, kw = get_positional(self.args, self.kw)
    if debug_:
      print("DEBUG:", self.f, x, args, kw)
    return self.f(x, *args, **kw)


class Composition(Composable):
  """A function composition.

  This simply composes its two argument functions when
  applied to a final argument via `of`.
  """

  def __init__(self, f, g):
    self.f = f
    self.g = g

  def funcall(self, x):
    return self.g.funcall(self.f.funcall(x))


# These are DSL names, not Python names
# pylint: disable=invalid-name, exec-used
def External(module_name, function_name):
  """Import a function from an external module.

  Note that the `module_name` must be a module name
  that works with the usual import mechanisms. Shorthands
  like "tf.nn" will not work.

  Args:
      module_name: name of the module
      function_name: name of the function within the module

  Returns:
      Function-wrapped value of symbol.
  """
  module = importlib.import_module(module_name)
  return Function(vars(module)[function_name])


def Import(statements):
  """Import a function by exec.

  Args:
      statements: Python statements

  Returns:
      Function-wrapped value of `f`.

  Raises:
      ValueError: the statements didn't define a value for "f"
  """
  environ = {}
  exec_(statements, environ)
  if "f" not in environ:
    raise ValueError("failed to define \"f\": %s", statements)
  f = environ["f"]
  return Function(f)


# pylint: enable=invalid-name, exec-used
def debug(mode=True):
  """Turn on/off debugging mode.

  Debugging mode prints more information about the construction
  of a network.

  Args:
      mode: True if turned on, False otherwise
  """
  global debug_
  debug_ = mode
