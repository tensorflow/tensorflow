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
"""Handles function calls, by generating compiled function names and calls.

Note: this transformer does not rename the top level object being converted;
that is the caller's responsibility.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct import templates


class FunctionNamer(object):
  """Describes the interface for CallTreeTransformer's namer."""

  def compiled_function_name(self,
                             original_name,
                             live_object=None,
                             owner_type=None):
    """Generate the name corresponding to the compiled version of a function.

    Args:
      original_name: String
      live_object: Callable, the actual target function, if known.
      owner_type: Optional object. If present, it indicates that the function is
          a member of the given type.
    Returns:
      String.
    """
    raise NotImplementedError()

  def compiled_class_name(self, original_name, live_object=None):
    """Generate the name corresponding to the compiled version of a class.

    Args:
      original_name: String
      live_object: The actual target class, if known.
    Returns:
      String.
    """
    raise NotImplementedError()


class CallTreeTransformer(gast.NodeTransformer):
  """Transforms the call tree by renaming transformed symbols."""

  def __init__(self, namer, namespace, uncompiled_modules,
               nocompile_decorators):
    self.namer = namer
    self.namespace = namespace
    self.uncompiled_modules = uncompiled_modules
    self.nocompile_decorators = nocompile_decorators

  # pylint:disable=invalid-name

  def _resolve_name(self, node):
    if isinstance(node, gast.Call):
      return self._resolve_name(node.func)
    if isinstance(node, gast.Name):
      return self.namespace.get(node.id)
    if isinstance(node, gast.Attribute):
      parent = self._resolve_name(node.value)
      if parent is not None:
        return getattr(parent, node.attr)
      return None
    raise ValueError(node)

  def _try_resolve_target(self, node):
    """Works for methods of objects of known type."""
    if anno.hasanno(node, 'live_val'):
      return anno.getanno(node, 'live_val')
    if isinstance(node, gast.Attribute) and anno.hasanno(node, 'type'):
      member = getattr(anno.getanno(node, 'type'), node.attr)
      return member
    return None

  def _should_compile(self, node, fqn):
    for i in range(1, len(fqn)):
      if fqn[:i] in self.uncompiled_modules:
        return False

    # Check for local decorations
    if anno.hasanno(node, 'graph_ready'):
      return False

    # The decorators themselves are not to be converted.
    # If present, the decorators should appear as static functions.
    target_obj = self._try_resolve_target(node.func)
    if target_obj is not None:
      # This attribute is set by the decorator itself.
      # TODO(mdan): This may not play nicely with other wrapping decorators.
      if hasattr(target_obj, '__pyct_is_compile_decorator'):
        return False

      if target_obj in self.nocompile_decorators:
        return False

      # Inspect the target function decorators. If any include a @convert
      # or @graph_ready annotation, then they must be called as they are.
      # TODO(mdan): This may be quite heavy.
      # To parse and re-analize each function for every call site could be quite
      # wasteful. Maybe we could cache the parsed AST?
      try:
        target_node = parser.parse_object(target_obj).body[0]
      except TypeError:
        # Functions whose source we cannot access are compilable (e.g. wrapped
        # to py_func).
        return True

      for dec in target_node.decorator_list:
        decorator_fn = self._resolve_name(dec)
        if (decorator_fn is not None and
            decorator_fn in self.nocompile_decorators):
          return False

    return True

  def _rename_compilable_function(self, node):
    assert anno.hasanno(node.func, 'live_val')
    assert anno.hasanno(node.func, 'fqn')
    target_obj = anno.getanno(node.func, 'live_val')
    target_fqn = anno.getanno(node.func, 'fqn')

    if not self._should_compile(node, target_fqn):
      return node

    if anno.hasanno(node, 'is_constructor'):
      new_name = self.namer.compiled_class_name(
          '__'.join(target_fqn), live_object=target_obj)
    else:
      new_name = self.namer.compiled_function_name(
          '__'.join(target_fqn), live_object=target_obj)
    node.func = gast.Name(id=new_name, ctx=gast.Load(), annotation=None)
    return node

  def _rename_member_function_of_known_type(self, node):
    assert isinstance(node.func, gast.Attribute)

    type_fqn = anno.getanno(node.func, 'type_fqn')
    assert anno.hasanno(node.func, 'type')
    target_type = anno.getanno(node.func, 'type')

    if not self._should_compile(node, type_fqn):
      return node

    # TODO(mdan): We should not assume that the namer only needs the
    # member function name.
    method_name = node.func.attr
    method_object = getattr(target_type, method_name)
    new_name = self.namer.compiled_function_name(
        method_name, live_object=method_object, owner_type=target_type)
    if new_name != node.func.attr:
      # If a member function call is renamed, then the new function is no
      # longer bound to the target object. We then refactor the call from:
      #   foo.bar(...)
      # to:
      #   renamed_foo(bar, ...)
      # TODO(mdan): This risks causing duplication, if target_type is renamed.
      node.args = [node.func.value] + node.args
      node.func = gast.Name(new_name, gast.Load(), None)
    return node

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
    # TODO(mdan): Rename this annotation to 'graph_ready'
    anno.setanno(wrapper_def, 'skip_processing', True)

    return (wrapper_def, call_expr)

  def _function_is_compilable(self, target_obj):
    # TODO(mdan): This is just a placeholder. Implement.
    return not isinstance(target_obj, types.BuiltinFunctionType)

  def visit_Expr(self, node):
    if isinstance(node.value, gast.Call):
      if anno.hasanno(node.value.func, 'live_val'):
        target_obj = anno.getanno(node.value.func, 'live_val')
        if not self._function_is_compilable(target_obj):
          if anno.hasanno(node.value.func, 'fqn'):
            target_fqn = anno.getanno(node.value.func, 'fqn')
            if not self._should_compile(node.value, target_fqn):
              return node
            node = self._wrap_to_py_func_no_return(node.value)
            return node
      # Only the case of py_func with no return value is special.
      # Everything else is processed by visit_Call.
      self.visit(node.value)
    else:
      self.generic_visit(node)
    return node

  def visit_Call(self, node):
    # If the function is wrapped by one of the marker decorators,
    # consider it graph ready.
    if anno.hasanno(node.func, 'live_val'):
      target_obj = anno.getanno(node.func, 'live_val')
      if target_obj in self.nocompile_decorators:
        if len(node.args) < 1:
          raise ValueError(
              'Found call to decorator function "%s", but it had no arguments. '
              'A decorator needs at least an argument.')
        anno.setanno(node.args[0], 'graph_ready', True)

    self.generic_visit(node)
    if anno.hasanno(node.func, 'live_val'):
      target_obj = anno.getanno(node.func, 'live_val')
      if self._function_is_compilable(target_obj):
        node = self._rename_compilable_function(node)
      else:
        raise NotImplementedError('py_func with return values')
    elif anno.hasanno(node.func, 'type_fqn'):
      node = self._rename_member_function_of_known_type(node)
    else:
      raise NotImplementedError(
          'Member function call (of unknown type): %s.' % node.func.id)
    return node

  # pylint:enable=invalid-name


def transform(node, namer, namespace, uncompiled_modules, nocompile_decorators):
  """Transform function call to the compiled counterparts.

  Args:
    node: AST to transform.
    namer: FunctionNamer-like.
    namespace: Dict mapping symbol names to their corresponding live objects.
    uncompiled_modules: set of string tuples, each tuple represents the fully
        qualified name of a package containing functions that will not be
        compiled.
    nocompile_decorators: A tuple containing decorators to be stripped from
        functions during conversion.
  Returns:
    A tuple (node, new_names):
        node: The transformed AST
        new_names: set(string), containing any newly-generated names
  """
  transformer = CallTreeTransformer(namer, namespace, uncompiled_modules,
                                    nocompile_decorators)
  node = transformer.visit(node)
  return node
