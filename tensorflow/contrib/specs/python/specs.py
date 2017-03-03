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
"""Builder for TensorFlow models specified using specs_ops.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import inspect

from six import exec_
from tensorflow.contrib.specs.python import params_ops
from tensorflow.contrib.specs.python import specs_lib
from tensorflow.contrib.specs.python import specs_ops


def eval_params(params, environment=None):
  """Evaluates a parameter specification and returns the environment.

  Args:
      params: parameter assignments as a string
      environment: a dictionary of input bindings

  Returns:
      Environment with additional bindings created by
      executing `params`

  Raises:
      Exception: other exceptions raised during execution of `params`
  """
  specs_lib.check_keywords(params)
  bindings = {}
  if environment: bindings.update(environment)
  exec_(params, vars(params_ops), bindings)  # pylint: disable=exec-used
  return bindings


def eval_spec(spec, environment=None):
  """Evaluates a spec and returns the environment.

  This function allows you to use a spec to obtain multiple bindings
  in an environment. That is useful if you use the spec language to
  specify multiple components of a larger network, for example: "left
  = Cr(64, [5,5]); right = Fc(64)" Usually, you will want to use
  `create_net` or `create_net_fun` below.

  Args:
      spec: specification as a string
      environment: a dictionary of input bindings

  Returns:
      Environment with additional bindings created by spec.

  Raises:
      Exception: other exceptions raised during execution of `spec`

  """
  specs_lib.check_keywords(spec)
  bindings = {}
  if environment: bindings.update(environment)
  exec_(spec, vars(specs_ops), bindings)  # pylint: disable=exec-used
  return bindings


def create_net_fun(spec, environment=None):
  """Evaluates a spec and returns the binding of `net`.

  Specs are written in a DSL based on function composition.  A spec
  like `net = Cr(64, [3, 3])` assigns an object that represents a
  single argument function capable of creating a network to
  the variable `net`.

  Args:
      spec: specification as a string, ending with a `net = ...` statement
      environment: a dictionary of input bindings

  Returns:
      A callable that instantiates the `net` binding.

  Raises:
      ValueError: spec failed to create a `net`
      Exception: other exceptions raised during execution of `spec`

  """
  bindings = eval_spec(spec, environment)
  net = bindings.get("net", None)
  if net is None:
    raise ValueError("spec failed to create 'net': %s" % (spec,))
  return net.funcall


def create_net(spec, inputs, environment=None):
  """Evaluates a spec and creates a network instance given the inputs.

  Args:
      spec: specification as a string, ending with a `net = ...` statement
      inputs: input that `net` is applied to
      environment: a dictionary of input bindings

  Returns:
      A callable that instantiates the `net` binding.

  Raises:
      ValueError: spec failed to create a `net`
      Exception: other exceptions raised during execution of `spec`
  """
  return create_net_fun(spec, environment)(inputs)


class LocalImport(object):
  """A class that allows us to temporarily import something.

  Attributes:
      frame: the frame in which the context manager was invocked
      names: a dictionary containing the new bindings
      old: variable bindings that have been shadowed by the import
  """

  def __init__(self, names):
    """Create a context manager that binds the names in values.

    Args:
        names: A dictionary or module containing the bindings.
    """
    if not isinstance(names, dict):
      names = vars(names)
    self.names = names

  def __enter__(self):
    self.frame = inspect.currentframe()
    bindings = self.frame.f_back.f_globals
    self.old = {k: bindings.get(k, None) for k in self.names.keys()}
    bindings.update(self.names)

  def __exit__(self, some_type, value, traceback):
    del some_type, value, traceback
    bindings = self.frame.f_back.f_globals
    bindings.update(self.old)
    for k, v in self.old.items():
      if v is None: del bindings[k]
    del self.frame

ops = LocalImport(specs_ops)
