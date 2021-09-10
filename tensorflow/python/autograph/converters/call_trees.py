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

Requires function_scopes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.utils import ag_logging


# TODO(mdan): Rename to FunctionCallsTransformer.


class _Function(object):

  no_root = True

  def __init__(self):
    self.context_name = None


set_trace_warned = False


class _ArgTemplateBuilder(object):
  """Constructs a tuple representing the positional arguments in a call.

  Example (yes, it's legal Python 3):

      f(*args1, b, *args2, c, d)  ->  args1 + (b,) + args2 + (c, d)
  """

  def __init__(self):
    self._arg_accumulator = []
    self._argspec = []
    self._finalized = False

  def _consume_args(self):
    if self._arg_accumulator:
      self._argspec.append(
          gast.Tuple(elts=self._arg_accumulator, ctx=gast.Load()))
      self._arg_accumulator = []

  def add_arg(self, a):
    self._arg_accumulator.append(a)

  def add_stararg(self, a):
    self._consume_args()
    self._argspec.append(
        gast.Call(
            gast.Name(
                'tuple', ctx=gast.Load(), annotation=None, type_comment=None),
            args=[a],
            keywords=()))

  def finalize(self):
    self._consume_args()
    self._finalized = True

  def to_ast(self):
    assert self._finalized
    if self._argspec:
      result = self._argspec[0]
      for i in range(1, len(self._argspec)):
        result = gast.BinOp(result, gast.Add(), self._argspec[i])
      return result
    return gast.Tuple([], gast.Load())


class CallTreeTransformer(converter.Base):
  """Transforms the call tree by renaming transformed symbols."""

  def visit_Lambda(self, node):
    if not anno.hasanno(node, 'function_context_name'):
      # Lambda functions created during the conversion process have no
      # context manager.
      return self.generic_visit(node)
    with self.state[_Function] as fn_scope:
      fn_scope.context_name = anno.getanno(node, 'function_context_name')
      return self.generic_visit(node)

  def visit_FunctionDef(self, node):
    # Decorators and arg defaults are part of the outer scope.
    node.decorator_list = self.visit_block(node.decorator_list)
    node.args.defaults = self.visit_block(node.args.defaults)
    for i, d in enumerate(node.args.kw_defaults):
      if d is not None:
        node.args.kw_defaults[i] = self.visit(d)
    with self.state[_Function] as fn_scope:
      # Note: if the conversion process ever creates helper functions, this
      # assumption will no longer hold.
      assert anno.hasanno(node, 'function_context_name'), (
          'The function_scopes converter always creates a scope for functions.')
      fn_scope.context_name = anno.getanno(node, 'function_context_name')
      node.body = self.visit_block(node.body)
      if node.returns:
        node.returns = self.visit(node.returns)
      return node

  def visit_With(self, node):
    # Context manager calls (in node.items) are not converted.
    node.body = self.visit_block(node.body)
    return node

  def _args_to_tuple(self, node):
    """Ties together all positional and *arg arguments in a single tuple."""
    # TODO(mdan): We could rewrite this to just a call to tuple(). Maybe better?
    # For example for
    #   f(a, b, *args)
    # instead of writing:
    #   (a, b) + args
    # just write this?
    #   tuple(a, b, *args)
    builder = _ArgTemplateBuilder()
    for a in node.args:
      if isinstance(a, gast.Starred):
        builder.add_stararg(a.value)
      else:
        builder.add_arg(a)
    builder.finalize()
    return builder.to_ast()

  def _kwargs_to_dict(self, node):
    """Ties together all keyword and **kwarg arguments in a single dict."""
    if node.keywords:
      return gast.Call(
          gast.Name(
              'dict', ctx=gast.Load(), annotation=None, type_comment=None),
          args=(),
          keywords=node.keywords)
    else:
      return parser.parse_expression('None')

  def visit_Call(self, node):
    full_name = str(anno.getanno(node.func, anno.Basic.QN, default=''))
    function_context_name = self.state[_Function].context_name
    node = self.generic_visit(node)

    # TODO(mdan): Refactor converted_call as a 'Call' operator.

    # Calls to the internal 'ag__' module are never converted (though their
    # arguments might be).
    if full_name.startswith('ag__.'):
      return node

    # Calls to the function context manager (inserted by function_scopes) are
    # also safe.
    if full_name.startswith(function_context_name + '.'):
      return node

    # Calls to pdb.set_trace or ipdb.set_trace are never converted. We don't use
    # the normal mechanisms to bypass these literals because they are sensitive
    # to the frame they are being called from.
    # TODO(mdan): Generalize this to a "static allowlist" config.
    if full_name in ('pdb.set_trace', 'ipdb.set_trace', 'breakpoint'):
      global set_trace_warned
      if not set_trace_warned:
        # TODO(mdan): Update and shorten once available on tensorflow.org.
        ag_logging.warning(
            'Detected `pdb.set_trace()` in user code. The code'
            ' generated by AutoGraph is not optimized for step-by-step'
            ' debugging. See https://github.com/tensorflow/tensorflow/'
            'blob/master/tensorflow/python/autograph/g3doc/reference/'
            'debugging.md.')
        set_trace_warned = True
      return node

    if (full_name == 'print' and
        not self.ctx.user.options.uses(converter.Feature.BUILTIN_FUNCTIONS)):
      return node

    template = """
      ag__.converted_call(func, args, kwargs, function_ctx)
    """
    new_call = templates.replace_as_expression(
        template,
        func=node.func,
        args=self._args_to_tuple(node),
        kwargs=self._kwargs_to_dict(node),
        function_ctx=function_context_name)

    return new_call


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
  node = qual_names.resolve(node)

  node = CallTreeTransformer(ctx).visit(node)
  return node
