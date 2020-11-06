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
"""op_reg_gen: Generate op registration code from composite op code."""

# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast as ast

from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.framework import op_def_registry
from tensorflow.python.util import tf_inspect

_COMPOSITE_ARG_LIST = ['op_name', 'inputs', 'attrs', 'derived_attrs', 'outputs']


class OpRegGenImpl(transformer.CodeGenerator):
  """Visit the AST and generate C++ op registration functions."""

  def __init__(self, ctx):
    super(OpRegGenImpl, self).__init__(ctx)
    self.ctx = ctx

  def visit_Name(self, node):
    return node.id

  def visit_Constant(self, node):
    return node.value

  def visit_keyword(self, node):
    return node.arg, self.visit(node.value)

  def visit_List(self, node):
    return [self.visit(cst) for cst in node.elts]

  def visit_arguments(self, node):
    return [self.visit(arg) for arg in node.args]

  def visit_FunctionDef(self, node):
    # TODO(fengliuai): create one utility method to match different apis and
    # shared it with the tfr_gen.py module.
    compose_dec = []
    for dec in node.decorator_list:
      if isinstance(dec, ast.Call):
        if isinstance(dec.func, ast.Attribute) and dec.func.attr == 'Composite':
          compose_dec.append(dec)
        if isinstance(dec.func, ast.Name) and dec.func.id == 'Composite':
          compose_dec.append(dec)

    if not compose_dec:
      # skip a non-composition function
      return
    elif len(compose_dec) > 1:
      raise KeyError('More than one TF ops decomposes for.')

    all_dec_args = {}
    for arg_name, arg_value in zip(_COMPOSITE_ARG_LIST, compose_dec[0].args):
      all_dec_args[arg_name] = self.visit(arg_value)

    kw_dec_args = dict([self.visit(kw) for kw in compose_dec[0].keywords])

    if all_dec_args.keys() & kw_dec_args.keys():
      raise KeyError('More arguments than expected.')

    all_dec_args.update(kw_dec_args)

    op_name = all_dec_args['op_name']
    op_def = op_def_registry.get(op_name)
    if op_def:
      if len(all_dec_args) > 1:
        # Op has been registered, so it is a user error to specify op def.
        raise ValueError('Op has been registered: ' + op_name)
      else:
        # Op has been registered, then we don't need to generate register code.
        return

    # Validates the function inputs match what are in the decorator.
    inputs = all_dec_args.get('inputs', [])
    attrs = all_dec_args.get('attrs', [])
    expected_args = [arg.split(':')[0] for arg in inputs + attrs]
    all_func_args = self.visit(node.args)

    if len(expected_args) != len(all_func_args):
      raise KeyError('Composition arguments do not match the registration.')

    cxx_reg_code = '\nREGISTER_OP("{0}")'.format(op_name)
    for input_ in inputs:
      cxx_reg_code += '\n    .Input("{0}")'.format(input_)
    for attr in attrs:
      py_str = attr.replace('"', '\'')
      cxx_reg_code += '\n    .Attr("{0}")'.format(py_str)
    for attr in all_dec_args.get('derived_attrs', []):
      py_str = attr.replace('"', '\'')
      cxx_reg_code += '\n    .Attr("{0}")'.format(py_str)
    for output_ in all_dec_args.get('outputs', []):
      cxx_reg_code += '\n    .Output("{0}")'.format(output_)
    cxx_reg_code += ';\n'
    self.emit(cxx_reg_code)


class OpRegGen(transpiler.GenericTranspiler):
  """Transforms Python objects into TFR MLIR source code."""

  def transform_ast(self, node, ctx):
    gen = OpRegGenImpl(ctx)
    gen.visit(node)
    return gen.code_buffer


def op_reg_gen(func):
  """Parse a function and emit the TFR functions."""
  op_reg_code, _ = OpRegGen().transform(func, None)
  return op_reg_code


def gen_register_op(source, method_prefix=None):
  """Parse a python code and emit the TFR functions from a target class."""
  mlir_funcs = [
      op_reg_gen(func)
      for name, func in tf_inspect.getmembers(source, tf_inspect.isfunction)
      if not method_prefix or name.startswith(method_prefix)
  ]
  headers = r"""
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
  """
  code = '\n'.join(mlir_funcs)
  return headers + code + '}  // namespace tensorflow\n'
