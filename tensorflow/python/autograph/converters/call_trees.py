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

from collections import namedtuple

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.util import tf_inspect


class FunctionInfo(namedtuple('FunctionInfo', ('dtype',))):
  pass


# TODO(mdan): Move this to a separate transformer.
KNOWN_NUMPY_FUNCTIONS = {
    ('numpy', 'random', 'binomial'): FunctionInfo(dtype='tf.int64'),
}


# TODO(mdan): Get rid of these interfaces. Can now depend directly on Namer.


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


# TODO(mdan): Rename to CallsTransformer.


class CallTreeTransformer(converter.Base):
  """Transforms the call tree by renaming transformed symbols."""

  def _resolve_decorator_name(self, node):
    """Used to resolve decorator info."""
    if isinstance(node, gast.Call):
      return self._resolve_decorator_name(node.func)
    if isinstance(node, gast.Name):
      # TODO(mdan): Add test coverage for this branch.
      return self.ctx.info.namespace.get(node.id)
    if isinstance(node, gast.Attribute):
      parent = self._resolve_decorator_name(node.value)
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
        # TODO(mdan): We should probably return None here rather than an error.
        raise ValueError('Type "%s" has no attribute "%s". Is it dynamic?' %
                         (owner_type, node.attr))
    return None

  def _function_is_compilable(self, target_entity):
    """Determines whether an entity can be compiled at all."""
    # TODO(mdan): Expand.
    if target_entity.__module__ is None:
      # Functions like builtins and NumPy don't expose a module.
      # Those in general should not be compiled.
      return False
    if inspect_utils.isbuiltin(target_entity):
      return False
    return True

  def _should_compile(self, node, fqn):
    """Determines whether an entity should be compiled in the context."""
    # TODO(mdan): Needs cleanup. We should remove the use of fqn altogether.
    module_name = fqn[0]
    for mod in self.ctx.program.uncompiled_modules:
      if module_name.startswith(mod[0] + '.'):
        return False

    for i in range(1, len(fqn)):
      if fqn[:i] in self.ctx.program.uncompiled_modules:
        return False

    target_entity = self._try_resolve_target(node.func)

    if target_entity is not None:

      # This may be reached when "calling" a callable attribute of an object.
      # For example:
      #
      #   self.fc = tf.keras.layers.Dense()
      #   self.fc()
      #
      for mod in self.ctx.program.uncompiled_modules:
        if target_entity.__module__.startswith(mod[0] + '.'):
          return False

      # Inspect the target function decorators. If any include a @convert
      # or @do_not_convert annotation, then they must be called as they are.
      # TODO(mdan): This may be quite heavy. Perhaps always dynamically convert?
      # To parse and re-analyze each function for every call site could be quite
      # wasteful. Maybe we could cache the parsed AST?
      try:
        target_node, _ = parser.parse_entity(target_entity)
        target_node = target_node.body[0]
      except TypeError:
        # Functions whose source we cannot access are compilable (e.g. wrapped
        # to py_func).
        return True

      # This attribute is set when the decorator was applied before the
      # function was parsed. See api.py.
      if hasattr(target_entity, '__ag_compiled'):
        return False

      for dec in target_node.decorator_list:
        decorator_fn = self._resolve_decorator_name(dec)
        if (decorator_fn is not None and
            decorator_fn in self.ctx.program.options.strip_decorators):
          return False

    return True

  def _rename_compilable_function(self, node):
    assert anno.hasanno(node.func, 'live_val')
    assert anno.hasanno(node.func, 'fqn')
    target_entity = anno.getanno(node.func, 'live_val')
    target_fqn = anno.getanno(node.func, 'fqn')

    if anno.hasanno(node, 'is_constructor'):
      new_name = self.ctx.namer.compiled_class_name(
          target_fqn, live_entity=target_entity)
      do_rename = True
    else:
      if anno.hasanno(node.func, 'parent_type'):
        owner_type = anno.getanno(node.func, 'parent_type')
      else:
        # Fallback - not reliable.
        owner_type = inspect_utils.getmethodclass(target_entity)
      new_name, do_rename = self.ctx.namer.compiled_function_name(
          target_fqn, live_entity=target_entity, owner_type=owner_type)

    if do_rename:
      if target_entity is not None:
        if tf_inspect.ismethod(target_entity):
          # The renaming process will transform it into a regular function.
          # TODO(mdan): Is this complete? How does it work with nested members?
          node.args = [node.func.value] + node.args
      node.func = templates.replace_as_expression(
          'func_name', func_name=new_name)
    return node

  def _wrap_to_py_func_single_return(self, node, dtype):
    # TODO(mdan): Properly handle varargs, etc.
    template = """
      ag__.utils.wrap_py_func(func, dtype, (args,), kwargs, False)
    """
    return templates.replace_as_expression(
        template,
        func=node.func,
        dtype=parser.parse_expression(dtype),
        args=node.args,
        kwargs=ast_util.keywords_to_dict(node.keywords))

  def _insert_dynamic_conversion(self, node):
    """Inlines a dynamic conversion for a dynamic function."""
    # TODO(mdan): Pass information on the statically compiled functions.
    # Having access to the statically compiled functions can help avoid
    # unnecessary compilation.
    # For example, this would lead to function `a` being compiled twice:
    #
    #   def a():
    #     v = b
    #     b()
    #   def b():
    #     a()
    #
    # This is really a problem with recursive calls, which currently can
    # only be gated by a static condition, and should be rare.
    # TODO(mdan): It probably makes sense to use dynamic conversion every time.
    # Before we could convert all the time though, we'd need a reasonable
    # caching mechanism.
    template = """
      ag__.converted_call(func, owner, options, args)
    """
    if isinstance(node.func, gast.Attribute):
      func = gast.Str(node.func.attr)
      owner = node.func.value
    else:
      func = node.func
      owner = parser.parse_expression('None')
    new_call = templates.replace_as_expression(
        template,
        func=func,
        owner=owner,
        options=self.ctx.program.options.to_ast(
            self.ctx.info.namespace,
            internal_convert_user_code=self.ctx.program.options.recursive),
        args=node.args)
    # TODO(mdan): Improve the template mechanism to better support this.
    new_call.keywords = node.keywords
    return new_call

  def _visit_decorators(self, decorator_list):
    if not self.ctx.program.options.uses(converter.Feature.DECORATORS):
      # When not processing decorators, strip everything that is encountered.
      return []

    return self.visit_block(decorator_list)

  def visit_FunctionDef(self, node):
    node.args = self.visit(node.args)
    node.body = self.visit_block(node.body)
    node.decorator_list = self._visit_decorators(node.decorator_list)
    node.returns = self.visit_block(node.returns)
    return node

  def visit_Call(self, node):
    if anno.hasanno(node.func, 'live_val'):
      target_entity = anno.getanno(node.func, 'live_val')

      if anno.hasanno(node.func, 'fqn'):
        target_fqn = anno.getanno(node.func, 'fqn')
      else:
        target_fqn = None

      if self._function_is_compilable(target_entity):
        if self._should_compile(node, target_fqn):
          node = self._rename_compilable_function(node)
        else:
          node = self.generic_visit(node)
          return node

      elif target_fqn and target_fqn in KNOWN_NUMPY_FUNCTIONS:
        # TODO(mdan): Should we replace these with equivalent TF ops instead?
        node = self._wrap_to_py_func_single_return(
            node, KNOWN_NUMPY_FUNCTIONS[target_fqn].dtype)

      elif inspect_utils.isbuiltin(target_entity):
        # Note: Any builtin that passed the builtins converter is assumed to be
        # safe for graph mode.
        return node

      else:
        raise NotImplementedError(
            'py_func with return values (unknown function)')
    else:
      # Special cases
      # TODO(mdan): These need a systematic review - there may be more.

      # 1. super() calls - these are preserved. The class conversion mechanism
      # will ensure that they return the correct value.
      if ast_util.matches(node, 'super(_)'):
        return node

      # 2. super().method calls - these are preserved as well, when the
      # conversion processes the entire class.
      if (ast_util.matches(node, 'super(_)._(_)') and
          self.ctx.info.owner_type is not None):
        return node

      node = self._insert_dynamic_conversion(node)
    return node


def transform(node, ctx):
  """Transform function call to the compiled counterparts.

  Args:
    node: AST
    ctx: EntityContext
  Returns:
    A tuple (node, new_names):
        node: The transformed AST
        new_names: set(string), containing any newly-generated names
  """
  return CallTreeTransformer(ctx).visit(node)
