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
"""Random code generation for testing/fuzzing."""
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import string

import gast
import numpy as np

from tensorflow.python.autograph.pyct import templates


class NodeSampler(object):
  sample_map = None

  def sample(self):
    nodes, magnitudes = zip(*self.sample_map.items())
    return np.random.choice(
        nodes, p=np.array(magnitudes, dtype='float32') / np.sum(magnitudes))


class StatementSampler(NodeSampler):
  sample_map = dict((
      (gast.Assign, 10),
      (gast.Print, 1),
      (gast.If, 2),
      (gast.While, 2),
      (gast.For, 0),
  ))


class ExpressionSampler(NodeSampler):
  sample_map = dict((
      (gast.UnaryOp, 1),
      (gast.BinOp, 8),
      (gast.Name, 1),
      (gast.Call, 0),
  ))


class CompareSampler(NodeSampler):
  sample_map = dict((
      (gast.Eq, 1),
      (gast.NotEq, 1),
      (gast.Lt, 1),
      (gast.LtE, 1),
      (gast.Gt, 1),
      (gast.GtE, 1),
      (gast.Is, 1),
      (gast.IsNot, 1),
  ))


class BinaryOpSampler(NodeSampler):
  sample_map = dict((
      (gast.Add, 1),
      (gast.Sub, 1),
      (gast.Mult, 1),
      (gast.Div, 1),
      (gast.FloorDiv, 1),
      (gast.Mod, 1),
      (gast.Pow, 1),
  ))


class UnaryOpSampler(NodeSampler):
  sample_map = dict(((gast.USub, 1), (gast.UAdd, 0)))


class NameSampler(NodeSampler):
  sample_map = dict((
      ('new', 1),
      ('existing', 1),
  ))


N_CONTROLFLOW_STATEMENTS = 10
N_FUNCTIONDEF_STATEMENTS = 10


class CodeGenerator(object):
  """Generate random syntactically-valid Python ASTs."""

  def __init__(self, max_depth=3, depth=0):
    self.max_depth = max_depth
    self.depth = depth

  def generate_statement(self):
    """Generate a statement node, dispatching to the correct class method."""
    desired_node = StatementSampler().sample()
    self.depth += 1

    # Enforce some constraints on generating statements.
    # E.g., if statements need at least 3 readable variables.
    # If we fail to satisfy our constraints, draw another sample.
    if desired_node in (gast.While, gast.For, gast.If):
      if self.depth > self.max_depth:
        return self.generate_statement()

    # Go get the generator method and run it
    method = 'generate_' + desired_node.__name__
    visitor = getattr(self, method)
    node = visitor()
    self.depth -= 1
    return node

  def sample_node_list(self, low, high, generator):
    """Generate a list of statements of random length.

    Args:
      low: Fewest number of statements to generate.
      high: Highest number of statements to generate.
      generator: Function to call to generate nodes.

    Returns:
      A list of statements.
    """
    statements = []
    for _ in range(np.random.randint(low, high)):
      statements.append(generator())
    return statements

  def generate_Name(self, ctx=gast.Load()):
    variable_name = '_' + ''.join(
        random.choice(string.ascii_lowercase) for _ in range(4))
    return gast.Name(variable_name, ctx=ctx, annotation=None)

  def generate_BinOp(self):
    # TODO(alexbw): convert to generate_expression when we get to limit
    # expression depth.
    op = BinaryOpSampler().sample()()
    return gast.BinOp(self.generate_Name(), op, self.generate_Name())

  def generate_Compare(self):
    op = CompareSampler().sample()()
    return gast.Compare(self.generate_Name(), [op], [self.generate_Name()])

  def generate_UnaryOp(self):
    operand = self.generate_Name()
    op = UnaryOpSampler().sample()()
    return gast.UnaryOp(op, operand)

  def generate_expression(self):
    desired_node = ExpressionSampler().sample()
    # Go get the generator method and run it
    method = 'generate_' + desired_node.__name__
    generator = getattr(self, method)
    return generator()

  def generate_Assign(self):
    """Generate an Assign node."""
    # Generate left-hand side
    target_node = self.generate_Name(gast.Store())
    # Generate right-hand side
    value_node = self.generate_expression()
    # Put it all together
    node = gast.Assign(targets=[target_node], value=value_node)
    return node

  def generate_If(self):
    """Generate an If node."""
    test = self.generate_Compare()

    # Generate true branch statements
    body = self.sample_node_list(
        low=1,
        high=N_CONTROLFLOW_STATEMENTS // 2,
        generator=self.generate_statement)

    # Generate false branch statements
    orelse = self.sample_node_list(
        low=1,
        high=N_CONTROLFLOW_STATEMENTS // 2,
        generator=self.generate_statement)

    node = gast.If(test, body, orelse)
    return node

  def generate_While(self):
    """Generate a While node."""

    test = self.generate_Compare()
    body = self.sample_node_list(
        low=1, high=N_CONTROLFLOW_STATEMENTS, generator=self.generate_statement)
    orelse = []  # not generating else statements

    node = gast.While(test, body, orelse)
    return node

  def generate_Call(self):
    raise NotImplementedError

  def generate_Return(self):
    return gast.Return(self.generate_expression())

  def generate_Print(self):
    return templates.replace('print(x)', x=self.generate_expression())[0]

  def generate_FunctionDef(self):
    """Generate a FunctionDef node."""

    # Generate the arguments, register them as available
    arg_vars = self.sample_node_list(
        low=2, high=10, generator=lambda: self.generate_Name(gast.Param()))
    args = gast.arguments(arg_vars, None, [], [], None, [])

    # Generate the function body
    body = self.sample_node_list(
        low=1, high=N_FUNCTIONDEF_STATEMENTS, generator=self.generate_statement)
    body.append(self.generate_Return())
    fn_name = self.generate_Name().id
    node = gast.FunctionDef(fn_name, args, body, (), None)
    return node


def generate_random_functiondef():
  return CodeGenerator().generate_FunctionDef()
