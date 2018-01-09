# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Handles function calls, by generating compiled function names and calls."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import templates


class FunctionNamer(object):
  """Describes the interface for CallTreeTransformer's namer."""

  def compiled_function_name(self, original_name, live_object=None):
    """Generate the name corresponding to the compiled version of a function.

    Args:
      original_name: String
      live_object: Callable, the actual target function, if known.
    Returns:
      String.
    """
    raise NotImplementedError()


class CallTreeTransformer(gast.NodeTransformer):
  """Transforms the call tree by renaming transformed symbols."""

  def __init__(self, namer, uncompiled_modules):
    self.namer = namer
    self.uncompiled_modules = uncompiled_modules

  # pylint:disable=invalid-name

  def visit_FunctionDef(self, node):
    self.generic_visit(node)
    node.name = self.namer.compiled_function_name(node.name)
    return node

  def _rename_compilable_function(self, node):
    assert anno.hasanno(node.func, 'live_val')
    assert anno.hasanno(node.func, 'fqn')
    target_obj = anno.getanno(node.func, 'live_val')
    target_fqn = anno.getanno(node.func, 'fqn')

    fqn = ''
    for s in target_fqn:
      if fqn:
        fqn += '.'
      fqn += s
      if fqn in self.uncompiled_modules:
        return node

    new_name = self.namer.compiled_function_name(fqn, live_object=target_obj)
    node.func = gast.Name(id=new_name, ctx=gast.Load(), annotation=None)
    return node

  def _rename_member_function_of_known_type(self, node):
    target_fqn = anno.getanno(node.func, 'type_fqn')

    fqn = ''
    for s in target_fqn:
      if fqn:
        fqn += '.'
      fqn += s
      if fqn in self.uncompiled_modules:
        return node

    raise NotImplementedError('Member function call (of known type).')

  def _wrap_to_py_func_no_return(self, node):
    args_scope = anno.getanno(node, 'args_scope')
    # TODO(mdan): Properly handle varargs, kwargs, etc.
    args = tuple(gast.Name(n, gast.Load(), None) for n in args_scope.used)

    # pylint:disable=undefined-variable,unused-argument,function-redefined

    def template(call, wrapper, args):

      def wrapper(args):
        call(args)
        return 1

      tf.py_func(wrapper, [args], [tf.int64])

    # pylint:enable=undefined-variable,unused-argument,function-redefined

    wrapper_name = self.namer.compiled_function_name(node.func.id)
    wrapper_def, call_expr = templates.replace(
        template,
        call=node.func,
        wrapper=gast.Name(wrapper_name, gast.Load(), None),
        args=args)
    anno.setanno(call_expr.value, 'args_scope', args_scope)
    anno.setanno(wrapper_def, 'skip_processing', True)

    return (wrapper_def, call_expr)

  def visit_Expr(self, node):
    if isinstance(node.value, gast.Call):
      node = self._wrap_to_py_func_no_return(node.value)
    else:
      self.generic_visit(node)
    return node

  def visit_Call(self, node):
    self.generic_visit(node)
    if anno.hasanno(node.func, 'live_val'):
      target_obj = anno.getanno(node.func, 'live_val')
      if isinstance(target_obj, types.BuiltinFunctionType):
        raise NotImplementedError('py_func with return values')
      else:
        node = self._rename_compilable_function(node)
    elif anno.hasanno(node.func, 'type_fqn'):
      node = self._rename_member_function_of_known_type(node)
    else:
      raise NotImplementedError(
          'Member function call (of unknown type): %s.' % node.func.id)
    return node

  # pylint:enable=invalid-name


def transform(node, namer, uncompiled_modules):
  """Transform function call to the compiled counterparts.

  Args:
    node: AST to transform.
    namer: FunctionNamer-like.
    uncompiled_modules: set of string tuples, each tuple represents the fully
        qualified name of a package containing functions that will not be
        compiled.
  Returns:
    A tuple (node, new_names):
        node: The transformed AST
        new_names: set(string), containing any newly-generated names
  """
  transformer = CallTreeTransformer(namer, uncompiled_modules)
  node = transformer.visit(node)
  return node
