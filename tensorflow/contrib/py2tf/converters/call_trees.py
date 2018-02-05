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
from tensorflow.contrib.py2tf.pyct import transformer
from tensorflow.contrib.py2tf.pyct.static_analysis.annos import NodeAnno
from tensorflow.python.util import tf_inspect


class FunctionNamer(object):
  """Describes the interface for CallTreeTransformer's namer."""

  def compiled_function_name(self,
                             original_fqn,
                             live_entity=None,
                             owner_type=None):
    """Generate the name corresponding to the compiled version of a function.

    Args:
      original_fqn: string or tuple(string)
      live_entity: Callable, the actual target function, if known.
      owner_type: Optional object. If present, it indicates that the function is
          a member of the given type.
    Returns:
      string, bool
    """
    raise NotImplementedError()

  def compiled_class_name(self, original_fqn, live_entity=None):
    """Generate the name corresponding to the compiled version of a class.

    Args:
      original_fqn: string or tuple(string)
      live_entity: The actual target class, if known.
    Returns:
      string
    """
    raise NotImplementedError()


class CallTreeTransformer(transformer.Base):
  """Transforms the call tree by renaming transformed symbols."""

  def __init__(self, context, uncompiled_modules, nocompile_decorators):
    super(CallTreeTransformer, self).__init__(context)
    self.uncompiled_modules = uncompiled_modules
    self.nocompile_decorators = nocompile_decorators

  # pylint:disable=invalid-name

  def _resolve_name(self, node):
    if isinstance(node, gast.Call):
      return self._resolve_name(node.func)
    if isinstance(node, gast.Name):
      return self.context.namespace.get(node.id)
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
      owner_type = anno.getanno(node, 'type')
      if hasattr(owner_type, node.attr):
        return getattr(owner_type, node.attr)
      else:
        raise ValueError('Type "%s" has not attribute "%s". Is it dynamic?' %
                         (owner_type, node.attr))
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
    target_entity = self._try_resolve_target(node.func)
    if target_entity is not None:
      # This attribute is set by the decorator itself.
      # TODO(mdan): This may not play nicely with other wrapping decorators.
      if hasattr(target_entity, '__pyct_is_compile_decorator'):
        return False

      if target_entity in self.nocompile_decorators:
        return False

      # Inspect the target function decorators. If any include a @convert
      # or @graph_ready annotation, then they must be called as they are.
      # TODO(mdan): This may be quite heavy.
      # To parse and re-analize each function for every call site could be quite
      # wasteful. Maybe we could cache the parsed AST?
      try:
        target_node, _ = parser.parse_entity(target_entity)
        target_node = target_node.body[0]
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

  def _determine_function_owner(self, m):
    # TODO(mdan): The parent type should be known at analysis. Use that instead.
    if hasattr(m, 'im_class'):  # Python 2
      return m.im_class
    if hasattr(m, '__qualname__'):  # Python 3
      # Object attributes: should be bound to "self".
      if hasattr(m, '__self__'):
        return type(m.__self__)

      # Class attributes: should have the owner name in their namespace.
      qn = m.__qualname__.split('.')
      if len(qn) < 2:
        return None
      owner_name, func_name = qn[-2:]
      if func_name != m.__name__:
        raise ValueError('Inconsistent names detected '
                         '(__qualname__[1] = "%s", __name__ = "%s") for %s.' %
                         (func_name, m.__name__, m))
      if owner_name == '<locals>':
        return None
      if owner_name not in self.context.namespace:
        raise ValueError(
            'Could not resolve name "%s" while analyzing %s. Namespace:\n%s' %
            (owner_name, m, self.context.namespace))
      return self.context.namespace[owner_name]
    return None

  def _rename_compilable_function(self, node):
    assert anno.hasanno(node.func, 'live_val')
    assert anno.hasanno(node.func, 'fqn')
    target_entity = anno.getanno(node.func, 'live_val')
    target_fqn = anno.getanno(node.func, 'fqn')

    if not self._should_compile(node, target_fqn):
      return node

    if anno.hasanno(node, 'is_constructor'):
      new_name = self.context.namer.compiled_class_name(
          target_fqn, live_entity=target_entity)
      do_rename = True
    else:
      owner_type = self._determine_function_owner(target_entity)
      new_name, do_rename = self.context.namer.compiled_function_name(
          target_fqn, live_entity=target_entity, owner_type=owner_type)

    if do_rename:
      if target_entity is not None:
        if tf_inspect.ismethod(target_entity):
          # The renaming process will transform it into a regular function.
          # TODO(mdan): Is this complete? How does it work with nested members?
          node.args = [node.func.value] + node.args
      node.func = templates.replace('func_name', func_name=new_name)[0]
    return node

  def _wrap_to_py_func_no_return(self, node):
    args_scope = anno.getanno(node, NodeAnno.ARGS_SCOPE)
    # TODO(mdan): Properly handle varargs, kwargs, etc.
    template = """
      def wrapper(args):
        call(args)
        return 1
      tf.py_func(wrapper, [args], [tf.int64])
    """
    wrapper_def, call_expr = templates.replace(
        template,
        call=node.func,
        wrapper=self.context.namer.compiled_function_name(node.func.id)[0],
        args=tuple(args_scope.used))
    anno.setanno(call_expr.value, NodeAnno.ARGS_SCOPE, args_scope)
    # TODO(mdan): Rename this annotation to 'graph_ready'
    anno.setanno(wrapper_def, anno.Basic.SKIP_PROCESSING, True)

    return (wrapper_def, call_expr)

  def _function_is_compilable(self, target_entity):
    # TODO(mdan): This is just a placeholder. Implement.
    return not isinstance(target_entity, types.BuiltinFunctionType)

  def visit_Expr(self, node):
    if isinstance(node.value, gast.Call):
      if anno.hasanno(node.value.func, 'live_val'):
        target_entity = anno.getanno(node.value.func, 'live_val')
        if not self._function_is_compilable(target_entity):
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
      target_entity = anno.getanno(node.func, 'live_val')
      if target_entity in self.nocompile_decorators:
        if len(node.args) < 1:
          raise ValueError(
              'Found call to decorator function "%s", but it had no arguments. '
              'A decorator needs at least an argument.')
        anno.setanno(node.args[0], 'graph_ready', True)

    self.generic_visit(node)
    if anno.hasanno(node.func, 'live_val'):
      target_entity = anno.getanno(node.func, 'live_val')
      if self._function_is_compilable(target_entity):
        node = self._rename_compilable_function(node)
      else:
        raise NotImplementedError('py_func with return values')
    else:
      if self.context.recursive:
        raise NotImplementedError('Could not resolve target function.')
      else:
        # TODO(mdan): Double check. Is this reachable code?
        pass
    return node

  # pylint:enable=invalid-name


def transform(node, context, uncompiled_modules, nocompile_decorators):
  """Transform function call to the compiled counterparts.

  Args:
    node: AST to transform.
    context: An EntityContext object.
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
  t = CallTreeTransformer(context, uncompiled_modules, nocompile_decorators)
  node = t.visit(node)
  return node
