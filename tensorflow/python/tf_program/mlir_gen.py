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
"""mlir_gen: Generate mlir code from python code."""

# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast as ast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
import tensorflow.python.tf_program.pywrap_tfd as tfp
from tensorflow.python.types import core


class SymbolTable(object):
  """Symbol Table for python code."""

  def __init__(self):
    self.symbols = []
    self.enter_scope()

  def enter_scope(self):
    """Enter a new scope - at function level."""
    self.symbols.append({'types': {}, 'symbols': {}})
    self.curr_table = self.symbols[len(self.symbols) - 1]

  def insert_symbol(self, name, value):
    self.curr_table['symbols'][name] = value
    self.curr_table['types'][name] = value.getType()
    return value

  def insert_type(self, name, type_):
    self.curr_table['types'][name] = type_

  def exit_scope(self):
    self.symbols.pop()
    self.curr_table = self.symbols[len(self.symbols) - 1]

  def lookup(self, name):
    curr_idx = len(self.symbols) - 1
    while curr_idx >= 0 and (name not in self.symbols[curr_idx]['symbols']):
      curr_idx -= 1
    if curr_idx < 0:
      return None
    return self.symbols[curr_idx]['symbols'][name]

  def lookup_type(self, name):
    curr_idx = len(self.symbols) - 1
    while curr_idx >= 0 and (name not in self.symbols[curr_idx]['types']):
      curr_idx -= 1
    if curr_idx < 0:
      return None
    return self.symbols[curr_idx]['types'][name]

  def __repr__(self):
    s = '\n'.join(
        ' ' * idx * 2 + str(table) for idx, table in enumerate(self.symbols))
    return s


class ProcessType(ast.NodeVisitor):
  """Visit a node and return processed type Currently only visits annotations and gives their type.
  """

  def __init__(self, prog, ctx):
    self.prog = prog
    self.ctx = ctx

  def visit_Attribute(self, node):
    # Supported: core.Tensor
    value = self.visit(node.value)
    if value is None or not hasattr(value, node.attr):
      raise AttributeError(str(type(value)) + ' has no attribute ' + node.attr)
    attr = getattr(value, node.attr)

    if attr == core.Tensor:
      return tfp.UnrankedTensorType.get(tfp.IntegerType.get(self.prog.ctx, 32))
    return attr

  def visit_Name(self, node):
    if node.id == 'int':
      return tfp.IntegerType.get(self.prog.ctx, 32)
    if node.id == 'bool':
      return tfp.IntegerType.get(self.prog.ctx, 1)
    if node.id in self.ctx.info.namespace:
      return self.ctx.info.namespace[node.id]


class MLIRGen(ast.NodeVisitor):
  """Visit the AST and generate MLIR code Requires liveness, reading_definitions.
  """

  def __init__(self, ctx):
    self.ctx = ctx
    self.symbol_table = SymbolTable()
    self.prog = tfp.TFProgram()
    self.opbuilder = None

  def visit_block(self, block):
    return [self.visit(item) for item in block]

  def process_type(self, node):
    return ProcessType(self.prog, self.ctx).visit(node)

  def visit_Assign(self, node):
    value = self.visit(node.value)
    if isinstance(value, tuple):
      # If it is a tuple of values, assign one to each in targets
      # TODO: This currently is assuming that all elts in targets[0] are Name
      # objects. This might not be always True.
      for key, val in zip(node.targets[0].elts, value):
        self.symbol_table.insert_symbol(key.id, val)
    else:
      self.symbol_table.insert_symbol(node.targets[0].id, value)

  def visit_BinOp(self, node):
    left = self.visit(node.left)
    right = self.visit(node.right)
    if isinstance(node.op, ast.Sub):
      return tfp.Tf_SubOp.create(self.opbuilder, self.opbuilder.getUnknownLoc(),
                                 left, right).getResult(0)
    if isinstance(node.op, ast.Add):
      return tfp.Tf_AddV2Op.create(self.opbuilder,
                                   self.opbuilder.getUnknownLoc(), left,
                                   right).getResult(0)

  def visit_BoolOp(self, node):
    values = [self.visit(value) for value in node.values]
    if isinstance(node.op, ast.Or):
      return tfp.OrOp.create(self.opbuilder, self.opbuilder.getUnknownLoc(),
                             values).getResult(0)
    if isinstance(node.op, ast.And):
      return tfp.AndOp.create(self.opbuilder, self.opbuilder.getUnknownLoc(),
                              values).getResult(0)

  def visit_Call(self, node):
    func = self.visit(node.func)
    args = [self.visit(arg) for arg in node.args]
    callop = tfp.Tf_LegacyCallOp.create(self.opbuilder,
                                        self.opbuilder.getUnknownLoc(),
                                        func.getType().getResults(), args,
                                        func.getName())
    if callop.getNumResults() == 1:
      return callop[0]
    return tuple(callop.getResult(idx) for idx in range(callop.getNumResults()))

  def visit_Compare(self, node):
    left = self.visit(node.left)
    opb = self.opbuilder
    for op, right in zip(node.ops, node.comparators):
      if isinstance(op, ast.Eq):
        left = tfp.Tf_EqualOp.create(opb, opb.getUnknownLoc(), left,
                                     self.visit(right)).getResult(0)
      elif isinstance(op, ast.Lt):
        left = tfp.Tf_LessOp.create(opb, opb.getUnknownLoc(), left,
                                    self.visit(right)).getResult(0)
      elif isinstance(op, ast.LtE):
        left = tfp.Tf_LessEqualOp.create(opb, opb.getUnknownLoc(), left,
                                         self.visit(right)).getResult(0)
      elif isinstance(op, ast.Gt):
        left = tfp.Tf_GreaterOp.create(opb, opb.getUnknownLoc(), left,
                                       self.visit(right)).getResult(0)
      elif isinstance(op, ast.GtE):
        left = tfp.Tf_GreaterEqualOp.create(opb, opb.getUnknownLoc(), left,
                                            self.visit(right)).getResult(0)
      elif isinstance(op, ast.NotEq):
        left = tfp.Tf_NotEqualOp.create(opb, opb.getUnknownLoc(), left,
                                        self.visit(right)).getResult(0)
      else:
        raise NotImplementedError('CompareOp operator not recognized')
    return left

  def visit_Constant(self, node):
    opb = self.opbuilder
    value = None
    if isinstance(node.value, int):
      value = tfp.Tf_ConstOp.create(
          opb, opb.getUnknownLoc(),
          tfp.IntegerAttr.get(
              tfp.IntegerType.get(self.prog.ctx, 32), node.value)).getResult(0)
    return value

  def visit_FunctionDef(self, node):
    # Cache the current builder
    cache_builder = self.opbuilder
    inputs, outputs = [], []

    for arg in node.args.args:
      inputs.append(self.process_type(arg.annotation))

    if node.returns:
      outputs = [self.process_type(node.returns)]

    currfunc = self.prog.add_function(
        self.ctx.namer.new_symbol(node.name, []),
        self.prog.get_function_type(inputs, outputs))

    # Add the function to symbol table and enter new scope
    self.symbol_table.insert_symbol(node.name, currfunc)
    self.symbol_table.enter_scope()

    # Add arguments to symbol table
    for arg, value in zip(node.args.args, currfunc.getArguments()):
      self.symbol_table.insert_symbol(arg.id, value)
    self.opbuilder = tfp.OpBuilder(currfunc.getBody())

    self.visit_block(node.body)
    self.symbol_table.exit_scope()
    self.opbuilder = cache_builder

  def visit_If(self, node):
    cond = self.visit(node.test)

    # Create ifop
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    orelse_scope = anno.getanno(node, annos.NodeAnno.ORELSE_SCOPE)
    modified_in_cond = list(body_scope.modified | orelse_scope.modified)
    outputs = [
        self.symbol_table.lookup_type(str(var)) for var in modified_in_cond
    ]
    ifop = tfp.IfOp.create(self.opbuilder, self.opbuilder.getUnknownLoc(), cond,
                           outputs)

    # Cache the builder
    cache_builder = self.opbuilder

    # Visit body
    self.opbuilder = tfp.OpBuilder(ifop.getRegion(0))
    # Enter scope to avoid values generated inside the region to come in symbol
    # table
    self.symbol_table.enter_scope()
    for stmt in node.body:
      self.visit(stmt)
    retvals = [
        self.symbol_table.lookup(str(varname)) for varname in modified_in_cond
    ]
    tfp.ReturnOp.create(self.opbuilder, self.opbuilder.getUnknownLoc(), retvals)
    self.symbol_table.exit_scope()

    # Visit orelse
    self.opbuilder = tfp.OpBuilder(ifop.getRegion(1))
    self.symbol_table.enter_scope()
    for stmt in node.orelse:
      self.visit(stmt)
    retvals = [
        self.symbol_table.lookup(str(varname)) for varname in modified_in_cond
    ]
    tfp.ReturnOp.create(self.opbuilder, self.opbuilder.getUnknownLoc(), retvals)
    self.symbol_table.exit_scope()

    # Reset builder and enter return values in symbol table
    self.opbuilder = cache_builder
    for idx, var in enumerate(modified_in_cond):
      self.symbol_table.insert_symbol(str(var), ifop.getResult(idx))

    if ifop.getNumResults() == 1:
      return ifop.getResult(0)

    return tuple(ifop.getResult(i) for i in range(ifop.getNumResults()))

  def visit_Name(self, node):
    if self.symbol_table.lookup(node.id):
      return self.symbol_table.lookup(node.id)
    raise NotImplementedError('Symbol not found' + node.id)

  def visit_Return(self, node):
    opb = self.opbuilder
    value = self.visit(node.value)
    if isinstance(value, tuple):
      # For more than one return values
      return tfp.ReturnOp.create(opb, opb.getUnknownLoc(), list(value))
    return tfp.ReturnOp.create(opb, opb.getUnknownLoc(), [value])

  def visit_Tuple(self, node):
    return tuple(self.visit(elt) for elt in node.elts)

  def visit_UnaryOp(self, node):
    operand = self.visit(node.operand)
    if isinstance(node.op, ast.USub):
      return tfp.Tf_NegOp.create(self.opbuilder, self.opbuilder.getUnknownLoc(),
                                 operand).getResult(0)

  def _get_basic_loop_vars(self, modified, live_in, live_out):
    # [This is directly from
    # tensorflow/python/autograph/converters/control_flow.py]
    # The loop variables corresponding to simple symbols (e.g. `x`).
    basic_loop_vars = []
    for s in modified:
      if s.is_composite():
        # TODO: Raise an error when this happens for a TF loop.
        continue
      # Variables not live into or out of the loop are considered local to the
      # loop.
      if s not in live_in and s not in live_out:
        continue
      basic_loop_vars.append(s)
    return frozenset(basic_loop_vars)

  def _get_composite_loop_vars(self, modified, live_in):
    # [This is directly from
    # tensorflow/python/autograph/converters/control_flow.py]
    # The loop variables corresponding to composite symbols (e.g. `self.x`).
    composite_loop_vars = []
    for s in modified:
      if not s.is_composite():
        continue
      # Mutations made to objects created inside the loop will appear as writes
      # to composite symbols. Because these mutations appear as modifications
      # made to composite symbols, we check whether the composite's parent is
      # actually live into the loop.
      # Example:
      #   while cond:
      #     x = Foo()
      #     x.foo = 2 * x.foo  # x.foo is live into the loop, but x is not.
      #
      # Note that some parents might not be symbols - for example, in x['foo'],
      # 'foo' is a parent, but it's a literal, not a symbol. We don't check the
      # liveness of literals.
      support_set_symbols = tuple(
          sss for sss in s.support_set if sss.is_symbol())
      if not all(sss in live_in for sss in support_set_symbols):
        continue
      composite_loop_vars.append(s)
    return frozenset(composite_loop_vars)

  def _get_loop_vars(self, node, modified):
    # [This is directly from python/autograph/converters/control_flow.py]
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
    live_in = anno.getanno(node, anno.Static.LIVE_VARS_IN)
    live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
    reserved_symbols = body_scope.referenced

    basic_loop_vars = self._get_basic_loop_vars(modified, live_in, live_out)
    composite_loop_vars = self._get_composite_loop_vars(modified, live_in)
    loop_vars = tuple(basic_loop_vars | composite_loop_vars)

    # Variable that are used or defined inside the loop, but not defined
    # before entering the loop. Only simple variables must be defined. The
    # composite ones will be implicitly checked at runtime.
    undefined_lives = basic_loop_vars - defined_in

    return loop_vars, reserved_symbols, undefined_lives

  def visit_While(self, node):

    # Create a new WhileOp
    # `inputs` are initial values for loop variables
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    loop_vars, _, _ = self._get_loop_vars(node, body_scope.modified)
    inputs = [self.symbol_table.lookup(str(name)) for name in loop_vars]
    types = [input_.getType() for input_ in inputs]
    while_op = tfp.WhileOp.create(self.opbuilder,
                                  self.opbuilder.getUnknownLoc(), inputs, types)

    # cache the current builder
    cache_builder = self.opbuilder

    # Process cond
    self.symbol_table.enter_scope()
    for input_, type_ in zip(loop_vars, types):
      self.symbol_table.insert_symbol(
          str(input_),
          while_op.getRegion(0).front().addArgument(type_))
    self.opbuilder = tfp.OpBuilder(while_op.getRegion(0))
    tfp.ReturnOp.create(self.opbuilder, self.opbuilder.getUnknownLoc(),
                        [self.visit(node.test)])
    self.symbol_table.exit_scope()

    # Process body
    self.symbol_table.enter_scope()
    for input_, type_ in zip(loop_vars, types):
      self.symbol_table.insert_symbol(
          str(input_),
          while_op.getRegion(1).front().addArgument(type_))
    self.opbuilder = tfp.OpBuilder(while_op.getRegion(1))
    self.visit_block(node.body)
    tfp.ReturnOp.create(
        self.opbuilder, self.opbuilder.getUnknownLoc(),
        [self.symbol_table.lookup(str(name)) for name in loop_vars])
    self.symbol_table.exit_scope()

    # Enter new values as symbols
    for idx, var in enumerate(loop_vars):
      self.symbol_table.insert_symbol(str(var), while_op.getResult(idx))

    # Restore builder
    self.opbuilder = cache_builder


def mlir_gen_internal(node, entity_info):
  """Returns mlir module for unprocessed node `node`."""
  namer = naming.Namer({})
  graphs = cfg.build(node)
  ctx = transformer.Context(entity_info, namer, None)
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx)
  node = reaching_definitions.resolve(node, ctx, graphs)
  node = reaching_fndefs.resolve(node, ctx, graphs)
  node = liveness.resolve(node, ctx, graphs)
  mlir_generator = MLIRGen(ctx)
  mlir_generator.visit(node)
  return mlir_generator.prog


def mlir_gen(func):
  """Parse a function and return TFProgram."""
  node, source = parser.parse_entity(func, future_features=())
  entity_info = transformer.EntityInfo(
      name=func.__name__,
      source_code=source,
      source_file=None,
      future_features=(),
      namespace=inspect_utils.getnamespace(func))
  return mlir_gen_internal(node, entity_info)


def mlir_gen_from_source(source=None, src_file=None):
  """Parse a function as either a string or from a supplied file path and return a TFProgram.
  """
  if source is None:
    source = open(src_file).read()
  node = ast.parse(source)
  entity_info = transformer.EntityInfo(
      name='mlir_module',
      source_code=source,
      source_file=None,
      future_features=(),
      namespace={})
  return mlir_gen_internal(node, entity_info)
