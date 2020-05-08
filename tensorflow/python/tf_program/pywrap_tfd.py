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
'''
Intermediate between python bindings for MLIR and mlir generation for tensorflow
program. This passes most of the mlir classes as is, but adds a few new
operations and the basic structure for a tensorflow program
'''

# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.tf_program.mlir_wrapper import mlir_wrapper as mlir

# Class Definitions
OpBuilder = mlir.OpBuilder
Block = mlir.Block

# Types
Type = mlir.Type
IntegerType = mlir.IntegerType
FloatType = mlir.FloatType
RankedTensorType = mlir.RankedTensorType
UnrankedTensorType = mlir.UnrankedTensorType
IntegerAttr = mlir.IntegerAttr

# Standard Ops
ReturnOp = mlir.ReturnOp

# TF Dialect Ops
Tf_AnyOp = mlir.Tf_AnyOp
Tf_AddV2Op = mlir.Tf_AddV2Op
Tf_ConstOp = mlir.Tf_ConstOp
Tf_EqualOp = mlir.Tf_EqualOp
Tf_GreaterEqualOp = mlir.Tf_GreaterEqualOp
Tf_GreaterOp = mlir.Tf_GreaterOp
Tf_LegacyCallOp = mlir.Tf_LegacyCallOp
Tf_LessEqualOp = mlir.Tf_LessEqualOp
Tf_LessOp = mlir.Tf_LessOp
Tf_NegOp = mlir.Tf_NegOp
Tf_NotEqualOp = mlir.Tf_NotEqualOp
Tf_SubOp = mlir.Tf_SubOp

class IfOp:
  '''
  tfp.if(cond) ({body}, {orelse}) : type
  If `cond` is true, `body` is executed, otherwise `orelse` is executed
  '''
  @classmethod
  def create(cls, opb, loc, cond, outputs):
    state = mlir.OperationState(loc, "tfp.If")
    state.addOperands([cond])
    state.addTypes(outputs)
    state.addRegion().push_back(Block.new())  # body region
    state.addRegion().push_back(Block.new())  # orelse region
    return opb.createOperation(state)

class OrOp:
  '''
  tfp.Or(ops...)
  This is like tf.Any, except that the first dimension is opened into `ops`.
  Returns a tensor of 1-bit integers which is "Logical OR" of the coressponding
  elements in ops...
  '''
  @classmethod
  def create(cls, opb, loc, values):
    state = mlir.OperationState(loc, "tfp.Or")
    state.addTypes([
        UnrankedTensorType.get(IntegerType.get(1, opb.getContext()))])
    state.addOperands(values)
    return opb.createOperation(state)

class AndOp:
  '''
  tfp.And(ops...)
  This is like tf.All, except that the first dimension is opened to `ops`.
  Returns a tensor of 1-bit integers which is "Logical AND" of the coressponding
  elements in ops...
  '''
  @classmethod
  def create(cls, opb, loc, values):
    state = mlir.OperationState(loc, "tfp.And")
    state.addTypes([
        UnrankedTensorType.get(IntegerType.get(1, opb.getContext()))])
    state.addOperands(values)
    return opb.createOperation(state)

class WhileOp:
  '''
  tfp.While(init-vals, {
    ^bb1(cond-args):
      cond-region
      return cond
  }, {
    ^bb1(body-args):
      body-region
  })
  As long as `cond-region` returns a "true"-like value, the body-region
  is executed and the arguments are replaced by its return values for the next
  iteration
  '''
  @classmethod
  def create(cls, opb, loc, inputs, outputs):
    state = mlir.OperationState(loc, "tfp.While")
    state.addOperands(inputs)
    state.addTypes(outputs)
    state.addRegion().push_back(Block.new())  # cond region
    state.addRegion().push_back(Block.new())  # body region
    return opb.createOperation(state)

class TFProgram:
  '''
  Python wrap for a Tensorflow Program (essentially an mlir Module)
  '''
  def __init__(self):
    mlir.registerDialects()
    self.ctx = mlir.MLIRContext()
    self.builder = mlir.Builder(self.ctx)
    self.module = mlir.ModuleOp.create(mlir.UnknownLoc.get(self.ctx))
    self.curr_func = None

  def add_function(self, name, func_type):
    self.curr_func = mlir.FuncOp.create(
        mlir.UnknownLoc.get(self.ctx), name, func_type)
    self.module.push_back(self.curr_func)
    return self.curr_func

  def get_function_type(self, inputs, outputs):
    return self.builder.getFunctionType(inputs, outputs)

  def dump(self):
    self.module.dump()

  def __str__(self):
    return self.module.getAsStr()
