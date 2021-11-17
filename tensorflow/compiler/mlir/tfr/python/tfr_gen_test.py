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
"""Tests for `tfr_gen` module."""

# pylint: disable=missing-function-docstring

import sys

from tensorflow.compiler.mlir.python.mlir_wrapper import filecheck_wrapper as fw
from tensorflow.compiler.mlir.tfr.python import composite
from tensorflow.compiler.mlir.tfr.python.tfr_gen import tfr_gen_from_module as tfr_gen
from tensorflow.compiler.mlir.tfr.resources import gen_test_ops as test_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_array_ops as array_ops
from tensorflow.python.ops import gen_math_ops as math_ops
from tensorflow.python.platform import test


Composite = composite.Composite

#--- test fn for mlir location ---


@Composite('TestInputNOp')
def _tfr_loc_test(x):
  n = 10
  x_sum = x[0]
  for i in range(1, n):
    x_sum = math_ops.Add(x_sum, x[i])
  return x_sum


#--- test fn for tfr tensors ---


@composite.Composite('TestNoOp')
def _tfr_tensor_empty_arg():
  pass


@composite.Composite('TestIdentityOp')
def _tfr_tensor_tensor(x):
  return x


@composite.Composite('TestIdentityNOp')
def _tfr_tensor_tensor_list(x):
  return x


@composite.Composite('TestInputNOp')
def _tfr_tensor_tensor_list_get_elt(x):
  return x[1]


@composite.Composite('TestOutputNOp')
def _tfr_tensor_tensor_list_output(x):
  return [x, x]


@composite.Composite('TestTwoInputsOp')
def _tfr_tensor_tensor_list_split(x, y, pred):
  z, _ = array_ops.Split(axis=0, value=x, num_split=2)
  (y, pred)  # pylint: disable=pointless-statement
  return z


@composite.Composite('TestTwoOutputsOp')
def _tfr_tensor_two_output(x):
  z = array_ops.Split(axis=0, value=x, num_split=2)
  return z[0], z[1]


@composite.Composite('TestNumAttrsOp')
def _tfr_tensor_tensor_with_cst(x1, y1, x2, y2):
  x = array_ops.OneHot(
      indices=[0, 2, -1, x1], depth=y1, on_value=True, off_value=False)
  (x, x2, y2)  # pylint: disable=pointless-statement
  return

#--- test fn for scf control flow ---


@composite.Composite('TestTwoInputsOp')
def _tfr_control_flow_if(x, y, pred):
  if pred:
    return x
  else:
    return y


@composite.Composite('TestThreeInputsOp')
def _tfr_control_flow_nested_if(x, y, z, select):
  if select == 'x':
    return x
  elif select == 'y':
    return y
  else:
    return z


@composite.Composite('TestInputNOp')
def _tfr_control_flow_range_for(x):
  # TODO(fengliuai): use len(x) instead
  n = 10
  x_sum = x[0]
  for i in range(1, n):
    x_sum = math_ops.Add(x_sum, x[i])
  return x_sum


@composite.Composite('TestInputNOp')
def _tfr_control_flow_tensor_list_size(ins):
  n = len(ins)
  if n == 0:
    return array_ops.Const(value=[[0, 1], [2, 3]], dtype=dtypes.int64)
  else:
    return math_ops.AddN(ins)


#--- test fn for tf ops ---


@composite.Composite('TestComplexTFOp')
def _tfr_tf_ops_complex(lhs, rhs):
  left_padding, _ = array_ops.SplitV(
      value=lhs, size_splits=[rhs, -1], axis=0, num_split=2)
  _, right_padding = array_ops.SplitV(
      value=lhs, size_splits=[rhs, rhs], axis=1, num_split=2)
  return [left_padding, right_padding]


@composite.Composite('TestIdentityOp')
def _tfr_tf_ops_tensor(x):
  return array_ops.Identity(x)


@composite.Composite('TestTwoInputsOp')
def _tfr_tf_ops_tensors(x, y, pred):
  if pred:
    return math_ops.Add(x, y)
  else:
    return array_ops.Concat(0, [x, y])


@composite.Composite('TestInputNOp')
def _tfr_tf_ops_with_defaults(ins):
  return test_ops.TestTwoInputsOp(ins[0], ins[1])


#--- test fn for tfr attributes ---


@composite.Composite('TestNumAttrsOp')
def _tfr_attrs_num_type(x, y, x1, y1):
  # int
  z0 = [x, y]
  z1 = x == y
  z2 = x < y
  z3 = x <= y
  z4 = x > y
  z5 = x >= y
  z6 = x != y
  z7 = x + y
  z8 = x - y
  z8 += x
  z8 += 1
  (z0, z1, z2, z3, z4, z5, z6, z7, z8)  # pylint: disable=pointless-statement

  # float
  z9 = x1 > y1
  z10 = x1 + y1
  z11 = [x1, y1]
  (z9, z10, z11)  # pylint: disable=pointless-statement
  return


@composite.Composite('TestNonNumAttrsOp')
def _tfr_attrs_tfr_type(x, y, z):
  z1 = x == y
  z2 = x == 'test'
  z3 = y == z
  (z1, z2, z3)  # pylint: disable=pointless-statement
  return


#--- test fn for shapes ---


@composite.Composite('TestIdentityOp')
def _tfr_shapes(x):
  s1 = x.shape
  s3 = x.shape.as_list()

  for i in range(len(s3)):
    s3[i]  # pylint: disable=pointless-statement

  for i in range(1, len(s3), 2):
    s3[i]  # pylint: disable=pointless-statement

  s5 = array_ops.Shape(x)
  (s1, s3, s5)  # pylint: disable=pointless-statement
  return x


#--- test fn for nested functions ---


@composite.Composite('TestIdentityNOp')
def _tfr_temp_op(x):
  return x


@composite.Composite('TestIdentityOp')
def _tfr_temp_use_op(x):
  y = _tfr_temp_op([x])
  return y[0]

#--- test fn for quant built-ins ---


# pylint: disable=undefined-variable
@composite.Composite('TestIdentityOp')
def _tfr_quant_test(x):
  y = _tfr_quant_raw_data(x)
  s, z = _tfr_quant_qparam(x)
  s = _tfr_quant_scale_factor(1.0, [s, s])
  s = _tfr_quant_scale_factor(1.0, [s])
  y = math_ops.Sub(y, z)
  qmin, qmax = _tfr_quant_act_range('RELU', 1.0, 0)
  (qmin, qmax)  # pylint: disable=pointless-statement
  d = _tfr_quant_rescale(y, s, 0)
  e = math_ops.Cast(x=d, DstT=dtypes.int16)
  f = math_ops.Cast(x=e, DstT=dtypes.int8)
  return f


@composite.Composite('TestIdentityNOp')
def _tfr_quant_test_n(x):
  y = _tfr_quant_raw_data(x)
  return y


class TFRGenTestBase(test.TestCase):

  def _check_code(self, tfr_code, exp_tfr_code):
    return self.assertTrue(fw.check(str(tfr_code), exp_tfr_code), str(tfr_code))


class TFRGenTensorTest(TFRGenTestBase):
  """MLIR Generation Tests for MLIR TFR Program."""

  def test_tfr_loc(self):
    mlir_code = tfr_gen(sys.modules[__name__], '_tfr_loc', [test_ops])
    mlir_code_exp = r"""
      CHECK-LABEL: tfr.func @tf__test_input_n_op(%x: !tfr.tensor_list) -> (!tfr.tensor) {
      CHECK-NEXT:   %[[n:.*]] = arith.constant 10 : i64
      CHECK-SAME        loc("tfr_gen_test.py":%{{.*}}:6)
      CHECK-NEXT:   %[[cst:.*]] = arith.constant 0 : index
      CHECK-SAME        loc("tfr_gen_test.py":%[[sum_line:.*]]:10)
      CHECK-NEXT:   %[[elt:.*]] = tfr.get_element %x[%[[cst]]] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-SAME        loc("tfr_gen_test.py":%[[sum_line]]:10)
      CHECK-NEXT:   %[[cst_1:.*]] = arith.constant 1 : i64
      CHECK-SAME        loc("tfr_gen_test.py":%[[for_line:.*]]:2)
      CHECK-NEXT:   %[[begin:.*]] = arith.index_cast %[[cst_1]] : i64 to index
      CHECK-SAME        loc("tfr_gen_test.py":%[[for_line]]:2)
      CHECK-NEXT:   %[[end:.*]] = arith.index_cast %[[n]] : i64 to index
      CHECK-SAME        loc("tfr_gen_test.py":%[[for_line]]:2)
      CHECK-NEXT:   %[[step:.*]] = arith.constant 1 : index
      CHECK-SAME        loc("tfr_gen_test.py":%[[for_line]]:2)
      CHECK-NEXT:   %[[for_stmt:.*]] = scf.for %[[itr_1:.*]] = %[[begin]] to %[[end]] step %[[step]]
      CHECK-SAME:       iter_args(%[[it_arg:.*]] = %[[elt]]) -> (!tfr.tensor) {
      CHECK-NEXT:     %[[elt_1:.*]] = tfr.get_element %x[%itr_1] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-SAME        loc("tfr_gen_test.py":%[[add_line:.*]]:34)
      CHECK-NEXT:     %[[Add:.*]] = tfr.call @tf__add(%[[it_arg]], %[[elt_1]]) : (!tfr.tensor, !tfr.tensor) -> (!tfr.tensor)
      CHECK-SAME        loc("tfr_gen_test.py":%[[add_line]]:12)
      CHECK-NEXT:     scf.yield %[[Add]] : !tfr.tensor
      CHECK-SAME        loc(unknown)
      CHECK-NEXT:   }
      CHECK-SAME        loc("tfr_gen_test.py":%[[for_line]]:2)
      CHECK-NEXT:   %{{.*}} = arith.constant true
      CHECK-SAME        loc(unknown)
      CHECK-NEXT:   tfr.return %[[for_stmt]] : !tfr.tensor
      CHECK-SAME        loc(unknown)
      CHECK-NEXT: }
      CHECK-SAME        loc("tfr_gen_test.py":%{{def_line:.*}}:0)
    """
    self._check_code(mlir_code, mlir_code_exp)

  def test_tfr_tensors(self):
    mlir_code = tfr_gen(sys.modules[__name__], '_tfr_tensor', [test_ops])
    mlir_code_exp = r"""
      CHECK-LABEL: tfr.func @tf__test_no_op() -> () {
      CHECK-NEXT:    tfr.return
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_identity_op(%x: !tfr.tensor) -> (!tfr.tensor) {
      CHECK-NEXT: constant true
      CHECK-NEXT:    tfr.return %x : !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_identity_n_op(%x: !tfr.tensor_list) -> (!tfr.tensor_list) {
      CHECK-NEXT: constant true
      CHECK-NEXT:    tfr.return %x : !tfr.tensor_list
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_input_n_op(%x: !tfr.tensor_list) -> (!tfr.tensor) {
      CHECK-NEXT: constant true
      CHECK-NEXT: %[[index:.*]] = arith.constant 1 : index
      CHECK-NEXT: %[[sub:.*]] = tfr.get_element %x[%cst_1] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT: tfr.return %[[sub]] : !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_output_n_op(%x: !tfr.tensor) -> (!tfr.tensor_list) {
      CHECK-NEXT: constant true
      CHECK-NEXT: %[[list:.*]] = "tfr.build_list"(%x, %x) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
      CHECK-NEXT: tfr.return %[[list]] : !tfr.tensor_list
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_two_inputs_op(%x: !tfr.tensor, %y: !tfr.tensor, %pred: i1{tfr.name="pred",tfr.default=false}) -> (!tfr.tensor) {
      CHECK-NEXT: %[[cst:.*]] = arith.constant 0 : i64
      CHECK-NEXT: %[[cst_1:.*]] = arith.constant 2 : i64
      CHECK-NEXT: %[[cst_2:.*]] = "tfr.constant_tensor"(%[[cst]]) : (i64) -> !tfr.tensor
      CHECK-NEXT: %[[Split:.*]] = tfr.call @tf__split(%[[cst_2]], %x, %[[cst_1]]) : (!tfr.tensor, !tfr.tensor, i64) -> (!tfr.tensor_list)
      CHECK-NEXT: %[[cst_4:.*]] = arith.constant 0 : index
      CHECK-NEXT: %[[elt:.*]] = tfr.get_element %[[Split]][%idx] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT: %[[cst_5:.*]] = arith.constant 1 : index
      CHECK-NEXT: %[[elt_1:.*]] = tfr.get_element %[[Split]][%idx_1] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT: constant true
      CHECK-NEXT: tfr.return %[[elt]] : !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_two_outputs_op(%x: !tfr.tensor) -> (!tfr.tensor, !tfr.tensor) {
      CHECK-NEXT: %[[cst:.*]] = arith.constant 0 : i64
      CHECK-NEXT: %[[cst_1:.*]] = arith.constant 2 : i64
      CHECK-NEXT: %[[cst_2:.*]] = "tfr.constant_tensor"(%[[cst]]) : (i64) -> !tfr.tensor
      CHECK-NEXT: %[[Split:.*]] = tfr.call @tf__split(%[[cst_2]], %x, %[[cst_1]]) : (!tfr.tensor, !tfr.tensor, i64) -> (!tfr.tensor_list)
      CHECK-NEXT: constant true
      CHECK-NEXT: %[[cst_4:.*]] = arith.constant 0 : index
      CHECK-NEXT: %[[elt:.*]] = tfr.get_element %[[Split]][%cst_4] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT: %[[cst_5:.*]] = arith.constant 1 : index
      CHECK-NEXT: %[[elt_1:.*]] = tfr.get_element %[[Split]][%cst_5] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT: tfr.return %[[elt]], %[[elt_1]] : !tfr.tensor, !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_num_attrs_op(%x1: i64{tfr.name="x1",tfr.default=-10}, %y1: i64{tfr.name="y1",tfr.default=1}, %x2: f32{tfr.name="x2",tfr.default=0.0}, %y2: f32{tfr.name="y2",tfr.default=-3.0}) -> () {
      CHECK-NEXT: %[[cst:.*]] = arith.constant 0 : i64
      CHECK-NEXT: %[[cst_1:.*]] = arith.constant 2 : i64
      CHECK-NEXT: %[[cst_2:.*]] = arith.constant 1 : i64
      CHECK-NEXT: %[[zero:.*]] = arith.constant 0 : i64
      CHECK-NEXT: %[[cst_3:.*]] = arith.subi %zero, %cst_2 : i64
      CHECK-NEXT: %[[list:.*]] = "tfr.build_list"(%[[cst]], %[[cst_1]], %[[cst_3]], %x1) : (i64, i64, i64, i64) -> !tfr.attr
      CHECK-NEXT: %[[cst_4:.*]] = arith.constant true
      CHECK-NEXT: %[[cst_5:.*]] = arith.constant false
      CHECK-NEXT: %[[cst_6:.*]] = "tfr.constant_tensor"(%[[list]]) : (!tfr.attr) -> !tfr.tensor
      CHECK-NEXT: %[[cst_7:.*]] = "tfr.constant_tensor"(%y1) : (i64) -> !tfr.tensor
      CHECK-NEXT: %[[cst_8:.*]] = "tfr.constant_tensor"(%[[cst_4]]) : (i1) -> !tfr.tensor
      CHECK-NEXT: %[[cst_9:.*]] = "tfr.constant_tensor"(%[[cst_5]]) : (i1) -> !tfr.tensor
      CHECK-NEXT: %[[cst_10:.*]] = arith.constant -1 : i64
      CHECK-NEXT: %[[OneHot:.*]] = tfr.call @tf__one_hot(%[[cst_6]], %[[cst_7]], %[[cst_8]], %[[cst_9]], %[[cst_10]])
      CHECK-SAME:   (!tfr.tensor, !tfr.tensor, !tfr.tensor, !tfr.tensor, i64) -> (!tfr.tensor)
      CHECK-NEXT: constant true
      CHECK-NEXT: tfr.return
      CHECK-NEXT: }
    """
    self._check_code(mlir_code, mlir_code_exp)

  def test_tfr_control_flow(self):
    mlir_code = tfr_gen(sys.modules[__name__], '_tfr_control_flow', [test_ops])
    mlir_code_exp = r"""
      CHECK-LABEL: tfr.func @tf__test_two_inputs_op(%x: !tfr.tensor, %y: !tfr.tensor,
      CHECK-SAME:     %pred: i1{tfr.name="pred",tfr.default=false}) -> (!tfr.tensor) {
      CHECK-NEXT: %[[if:.*]] = scf.if %pred -> (!tfr.tensor) {
      CHECK-NEXT:   arith.constant true
      CHECK-NEXT:   scf.yield %x : !tfr.tensor
      CHECK-NEXT: } else {
      CHECK-NEXT:   arith.constant true
      CHECK-NEXT:   scf.yield %y : !tfr.tensor
      CHECK-NEXT:   }
      CHECK-NEXT:   tfr.return %if_stmt : !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_three_inputs_op(%x: !tfr.tensor, %y: !tfr.tensor, %z: !tfr.tensor,
      CHECK-SAME:     %select: !tfr.attr{tfr.name="act",tfr.default="z"}) -> (!tfr.tensor) {
      CHECK-NEXT:   %[[cst:.*]] = tfr.constant "x" -> !tfr.attr
      CHECK-NEXT:   %[[eq:.*]] = tfr.equal %select, %[[cst]] -> i1
      CHECK-NEXT:   %[[if_stmt:.*]] = scf.if %[[eq]] -> (!tfr.tensor) {
      CHECK-NEXT:     %[[cst_1:.*]] = arith.constant true
      CHECK-NEXT:     scf.yield %x : !tfr.tensor
      CHECK-NEXT:   } else {
      CHECK-NEXT:     %[[cst_2:.*]] = tfr.constant "y" -> !tfr.attr
      CHECK-NEXT:     %[[eq_1:.*]] = tfr.equal %select, %[[cst_2]] -> i1
      CHECK-NEXT:     %[[if_stmt1:.*]] = scf.if %[[eq_1]] -> (!tfr.tensor) {
      CHECK-NEXT:       %[[cst_3:.*]] = arith.constant true
      CHECK-NEXT:       scf.yield %y : !tfr.tensor
      CHECK-NEXT:     } else {
      CHECK-NEXT:       %[[cst_4:.*]] = arith.constant true
      CHECK-NEXT:       scf.yield %z : !tfr.tensor
      CHECK-NEXT:     }
      CHECK-NEXT:     scf.yield %[[if_stmt1]] : !tfr.tensor
      CHECK-NEXT:   }
      CHECK-NEXT:   tfr.return %[[if_stmt]] : !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_input_n_op(%x: !tfr.tensor_list) -> (!tfr.tensor) {
      CHECK-NEXT:   %[[n:.*]] = arith.constant 10 : i64
      CHECK-NEXT:   %[[cst:.*]] = arith.constant 0 : index
      CHECK-NEXT:   %[[elt:.*]] = tfr.get_element %x[%[[cst]]] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT:   %[[cst_1:.*]] = arith.constant 1 : i64
      CHECK-NEXT:   %[[begin:.*]] = arith.index_cast %[[cst_1]] : i64 to index
      CHECK-NEXT:   %[[end:.*]] = arith.index_cast %[[n]] : i64 to index
      CHECK-NEXT:   %[[step:.*]] = arith.constant 1 : index
      CHECK-NEXT:   %[[for_stmt:.*]] = scf.for %[[itr_1:.*]] = %[[begin]] to %[[end]] step %[[step]]
      CHECK-SAME:       iter_args(%[[it_arg:.*]] = %[[elt]]) -> (!tfr.tensor) {
      CHECK-NEXT:     %[[elt_1:.*]] = tfr.get_element %x[%itr_1] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT:     %[[Add:.*]] = tfr.call @tf__add(%[[it_arg]], %[[elt_1]]) : (!tfr.tensor, !tfr.tensor) -> (!tfr.tensor)
      CHECK-NEXT:     scf.yield %[[Add]] : !tfr.tensor
      CHECK-NEXT:   }
      CHECK-NEXT:   %{{.*}} = arith.constant true
      CHECK-NEXT:   tfr.return %[[for_stmt]] : !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_input_n_op(%ins: !tfr.tensor_list) -> (!tfr.tensor) {
      CHECK: %[[attr:.*]] = tfr.constant i64 -> !tfr.attr
      CHECK: %Const = tfr.call @tf__const(%{{.*}}, %[[attr]]) : (!tfr.attr, !tfr.attr) -> (!tfr.tensor)
    """
    self._check_code(mlir_code, mlir_code_exp)

  def test_tfr_tf_ops(self):
    mlir_code = tfr_gen(sys.modules[__name__], '_tfr_tf_ops', [test_ops])
    mlir_code_exp = r"""
      CHECK-LABEL: tfr.func @tf__test_complex_tf_op(%lhs: !tfr.tensor, %rhs: !tfr.tensor) -> (!tfr.tensor_list) {
      CHECK-NEXT:   %[[cst:.*]] = arith.constant 1 : i64
      CHECK-NEXT:   %[[zero:.*]] = arith.constant 0 : i64
      CHECK-NEXT:   %[[cst_1:.*]] = arith.subi %[[zero]], %cst : i64
      CHECK-NEXT:   %[[cst_2:.*]] = "tfr.constant_tensor"(%[[cst_1]]) : (i64) -> !tfr.tensor
      CHECK-NEXT:   %[[list:.*]] = "tfr.build_list"(%rhs, %[[cst_2]]) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
      CHECK-NEXT:   %[[cst_3:.*]] = arith.constant 0 : i64
      CHECK-NEXT:   %[[cst_4:.*]] = arith.constant 2 : i64
      CHECK-NEXT:   %[[zero_1:.*]] = arith.constant 0 : i64
      CHECK-NEXT:   %[[pack:.*]] = tfr.call @tf__pack(%[[list]], %[[zero_1]]) : (!tfr.tensor_list, i64) -> !tfr.tensor
      CHECK-NEXT:   %[[cst_5:.*]] = "tfr.constant_tensor"(%[[cst_3]]) : (i64) -> !tfr.tensor
      CHECK-NEXT:   %[[SplitV:.*]] = tfr.call @tf__split_v(%lhs, %[[pack]], %[[cst_5]], %[[cst_4]])
      CHECK-NEXT:   %[[idx:.*]] = arith.constant 0 : index
      CHECK-NEXT:   %[[elt:.*]] = tfr.get_element %SplitV[%idx] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT:   %[[idx_1:.*]] = arith.constant 1 : index
      CHECK-NEXT:   %[[elt_1:.*]] = tfr.get_element %SplitV[%idx_1] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT:   %[[list_1:.*]] = "tfr.build_list"(%rhs, %rhs) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
      CHECK-NEXT:   %[[cst_6:.*]] = arith.constant 1 : i64
      CHECK-NEXT:   %[[cst_7:.*]] = arith.constant 2 : i64
      CHECK-NEXT:   %[[zero_2:.*]] = arith.constant 0 : i64
      CHECK-NEXT:   %[[pack_1:.*]] = tfr.call @tf__pack(%[[list_1]], %[[zero_2]]) : (!tfr.tensor_list, i64) -> !tfr.tensor
      CHECK-NEXT:   %[[cst_8:.*]] = "tfr.constant_tensor"(%[[cst_6]]) : (i64) -> !tfr.tensor
      CHECK-NEXT:   %[[SplitV_1:.*]] = tfr.call @tf__split_v(%lhs, %[[pack_1]], %[[cst_8]], %[[cst_7]])
      CHECK-NEXT:   %[[idx_2:.*]] = arith.constant 0 : index
      CHECK-NEXT:   %[[elt_2:.*]] = tfr.get_element %SplitV_1[%idx_2] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT:   %[[idx_3:.*]] = arith.constant 1 : index
      CHECK-NEXT:   %[[elt_3:.*]] = tfr.get_element %SplitV_1[%idx_3] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT:   %[[cst_9:.*]] = arith.constant true
      CHECK-NEXT:   %[[list_2:.*]] = "tfr.build_list"(%[[elt]], %[[elt_3]]) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
      CHECK-NEXT:   tfr.return %[[list_2]] : !tfr.tensor_list
      CHECK-NEXT:   }

      CHECK-LABEL: tfr.func @tf__test_identity_op(%x: !tfr.tensor) -> (!tfr.tensor) {
      CHECK-NEXT:    %cst = arith.constant true
      CHECK-NEXT:    %[[Id:.*]] = tfr.call @tf__identity(%x) : (!tfr.tensor) -> (!tfr.tensor)
      CHECK-NEXT:    tfr.return %[[Id]] : !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_two_inputs_op(%x: !tfr.tensor, %y: !tfr.tensor,
      CHECK-SAME:     %pred: i1{tfr.name="pred",tfr.default=false}) -> (!tfr.tensor) {
      CHECK-NEXT:   %[[if_stmt:.*]] = scf.if %pred -> (!tfr.tensor) {
      CHECK-NEXT:     %cst = arith.constant true
      CHECK-NEXT:     %[[Add:.*]] = tfr.call @tf__add(%x, %y) : (!tfr.tensor, !tfr.tensor) -> (!tfr.tensor)
      CHECK-NEXT:     scf.yield %[[Add]] : !tfr.tensor
      CHECK-NEXT:   } else {
      CHECK-NEXT:     %cst_1 = arith.constant true
      CHECK-NEXT:     %[[cst_2:.*]] = arith.constant 0 : i64
      CHECK-NEXT:     %[[list:.*]] = "tfr.build_list"(%x, %y) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
      CHECK-NEXT:     %[[Concat:.*]] = tfr.call @tf__concat(%[[cst_2]], %[[list]]) : (i64, !tfr.tensor_list) -> (!tfr.tensor)
      CHECK-NEXT:     scf.yield %[[Concat]] : !tfr.tensor
      CHECK-NEXT:   }
      CHECK-NEXT:   tfr.return %[[if_stmt]] : !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_input_n_op(%ins: !tfr.tensor_list) -> (!tfr.tensor) {
      CHECK-NEXT:   %cst = arith.constant true
      CHECK-NEXT:   %[[cst_1:.*]] = arith.constant 0 : index
      CHECK-NEXT:   %[[elt:.*]] = tfr.get_element %ins[%cst_1] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT:   %[[cst_2:.*]] = arith.constant 1 : index
      CHECK-NEXT:   %[[elt_1:.*]] = tfr.get_element %ins[%cst_2] : (!tfr.tensor_list, index) -> !tfr.tensor
      CHECK-NEXT:   %[[cst_3:.*]] = arith.constant false
      CHECK-NEXT:   %[[call:.*]] = tfr.call @tf__test_two_inputs_op(
      CHECK-SAME:     %[[elt]], %[[elt_1]], %[[cst_3]]) : (!tfr.tensor, !tfr.tensor, i1) -> (!tfr.tensor)
      CHECK-NEXT:   tfr.return %[[call]] : !tfr.tensor
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__add_(!tfr.tensor<T>,!tfr.tensor<T>) -> (!tfr.tensor<T>) attributes {T,f32_,i1_,i32_,i64_}

      CHECK-LABEL: tfr.func @tf__concat_(!tfr.tensor<i32_>,!tfr.tensor_list<N,T>) -> (!tfr.tensor<T>) attributes {N,T,f32_,i1_,i32_,i64_}

      CHECK-LABEL: tfr.func @tf__identity_(!tfr.tensor<T>) -> (!tfr.tensor<T>) attributes {T,f32_,i1_,i32_,i64_}

      CHECK-LABEL: tfr.func @tf__pack_(!tfr.tensor_list<N,T>,i64{tfr.name="axis",tfr.type="int"}) -> (!tfr.tensor<T>) attributes {N,T,axis,f32_,i1_,i32_,i64_}

      CHECK-LABEL: tfr.func @tf__split_v_(!tfr.tensor<T>,!tfr.tensor<Tlen>,!tfr.tensor<i32_>,i64{tfr.name="num_split",tfr.type="int"}) -> (!tfr.tensor_list<num_split,T>) attributes {T,Tlen,f32_,i1_,i32_,i64_,num_split}

      CHECK-LABEL: tfr.func @tf__test_complex_tf_op_(!tfr.tensor<T>,!tfr.tensor<Tlen>,i64{tfr.name="N",tfr.type="int"}) -> (!tfr.tensor_list<N,T>) attributes {N,T,Tlen,f32_,i1_,i32_,i64_}

      CHECK-LABEL: tfr.func @tf__test_identity_op_(!tfr.tensor<T>) -> (!tfr.tensor<T>) attributes {T,f32_,i1_,i32_,i64_}

      CHECK-LABEL: tfr.func @tf__test_input_n_op_(!tfr.tensor_list<N,T>) -> (!tfr.tensor<T>) attributes {N,T,f32_,i1_,i32_,i64_}

      CHECK-LABEL: tfr.func @tf__test_two_inputs_op_(!tfr.tensor<T>,!tfr.tensor<T>,i1{tfr.name="pred",tfr.type="bool"}) -> (!tfr.tensor<T>) attributes {T,f32_,i1_,i32_,i64_,pred}

      CHECK-LABEL: tfr.func @tf__test_two_outputs_op_(!tfr.tensor<T>) -> (!tfr.tensor<T>,!tfr.tensor<T>) attributes {T,f32_,i1_,i32_,i64_}
    """
    self._check_code(mlir_code, mlir_code_exp)

  def test_tfr_attrs(self):
    mlir_code = tfr_gen(sys.modules[__name__], '_tfr_attrs', [test_ops])
    mlir_code_exp = r"""
      CHECK-LABEL: tfr.func @tf__test_num_attrs_op(
      CHECK-SAME:     %x: i64{tfr.name="x1",tfr.default=-10},
      CHECK-SAME:     %y: i64{tfr.name="y1",tfr.default=1},
      CHECK-SAME:     %x1: f32{tfr.name="x2",tfr.default=0.0},
      CHECK-SAME:     %y1: f32{tfr.name="y2",tfr.default=-3.0}) -> () {
      CHECK-NEXT: %{{.*}} = "tfr.build_list"(%x, %y) : (i64, i64) -> !tfr.attr
      CHECK-NEXT: %{{.*}} = arith.cmpi "eq", %x, %y : i64
      CHECK-NEXT: %{{.*}} = arith.cmpi "ult", %x, %y : i64
      CHECK-NEXT: %{{.*}} = arith.cmpi "ule", %x, %y : i64
      CHECK-NEXT: %{{.*}} = arith.cmpi "ugt", %x, %y : i64
      CHECK-NEXT: %{{.*}} = arith.cmpi "uge", %x, %y : i64
      CHECK-NEXT: %{{.*}} = arith.cmpi "ne", %x, %y : i64
      CHECK-NEXT: %{{.*}} = arith.addi %x, %y : i64
      CHECK-NEXT: %[[sub_1:.*]] = arith.subi %x, %y : i64
      CHECK-NEXT: %[[add_1:.*]] = arith.addi %[[sub_1]], %x : i64
      CHECK-NEXT: %[[cst:.*]] = arith.constant 1 : i64
      CHECK-NEXT: %{{.*}} = arith.addi %[[add_1]], %[[cst]] : i64
      CHECK-NEXT: %{{.*}} = arith.cmpf "ugt", %x1, %y1 : f32
      CHECK-NEXT: %{{.*}} = arith.addf %x1, %y1 : f32
      CHECK-NEXT: %{{.*}} = "tfr.build_list"(%x1, %y1) : (f32, f32) -> !tfr.attr
      CHECK-NEXT: %{{.*}} = arith.constant true
      CHECK-NEXT: tfr.return
      CHECK-NEXT: }

      CHECK-LABEL: tfr.func @tf__test_non_num_attrs_op(
      CHECK-SAME:     %x: !tfr.attr{tfr.name="z"},
      CHECK-SAME:     %y: !tfr.attr{tfr.name="x",tfr.default="hello"},
      CHECK-SAME:     %z: !tfr.attr{tfr.name="y",tfr.default=f32}) -> () {
      CHECK-NEXT: %{{.*}} = tfr.equal %x, %y -> i1
      CHECK-NEXT: %[[cst:.*]] = tfr.constant "test" -> !tfr.attr
      CHECK-NEXT: %{{.*}} = tfr.equal %x, %[[cst]] -> i1
      CHECK-NEXT: %{{.*}} = tfr.equal %y, %z -> i1
      CHECK-NEXT: %{{.*}} = arith.constant true
      CHECK-NEXT: tfr.return
      CHECK-NEXT: }
    """
    self._check_code(mlir_code, mlir_code_exp)

  def test_tf_tensor_shape(self):
    mlir_code = tfr_gen(sys.modules[__name__], '_tfr_shapes', [test_ops])
    mlir_code_exp = r"""
      CHECK-LABEL: tfr.func @tf__test_identity_op(%x: !tfr.tensor) -> (!tfr.tensor) {
      CHECK-NEXT:   %[[shape:.*]] = tfr.get_shape %x -> !shape.shape

      CHECK-NEXT:   %[[shape_1:.*]] = tfr.get_shape %x -> !shape.shape
      CHECK-NEXT:   %[[len:.*]] = shape.rank %[[shape_1]] : !shape.shape -> !shape.size
      CHECK-NEXT:   %[[index:.*]] = shape.size_to_index %[[len]] : !shape.size
      CHECK-NEXT:   %[[begin:.*]] = arith.constant 0 : index
      CHECK-NEXT:   %[[step:.*]] = arith.constant 1 : index
      CHECK-NEXT:   scf.for %[[itr_1:.*]] = %[[begin]] to %[[index]] step %[[step]]  {
      CHECK-NEXT:     %[[size:.*]] = shape.get_extent %[[shape_1]], %[[itr_1]]: !shape.shape, index -> !shape.size
      CHECK-NEXT:     %[[elt:.*]] = shape.size_to_index %[[size]] : !shape.size
      CHECK-NEXT:     scf.yield
      CHECK-NEXT:   }

      CHECK-NEXT:   %[[cst:.*]] = arith.constant 1 : i64
      CHECK-NEXT:   %[[len_1:.*]] = shape.rank %shape_1 : !shape.shape -> !shape.size
      CHECK-NEXT:   %[[len_size_1:.*]] = shape.size_to_index %[[len_1]] : !shape.size
      CHECK-NEXT:   %[[cst_1:.*]] = arith.constant 2 : i64
      CHECK-NEXT:   %[[begin_1:.*]] = arith.index_cast %[[cst]] : i64 to index
      CHECK-NEXT:   %[[step_1:.*]] = arith.index_cast %[[cst_1]] : i64 to index
      CHECK-NEXT:   scf.for %[[itr_3:.*]] = %[[begin_1]] to %[[len_size_1]] step %[[step_1]]

      CHECK:        %[[cst:.*]] = tfr.constant i32 -> !tfr.attr
      CHECK-NEXT:   %[[Shape:.*]] = tfr.call @tf__shape(%x, %[[cst]]) : (!tfr.tensor, !tfr.attr) -> (!tfr.tensor)
      CHECK-NEXT:   %{{.*}} = arith.constant true
      CHECK-NEXT:   tfr.return %x : !tfr.tensor
      CHECK-NEXT: }
    """
    self._check_code(mlir_code, mlir_code_exp)

  def test_temp_function(self):
    mlir_code = tfr_gen(sys.modules[__name__], '_tfr_temp', [test_ops])
    mlir_code_exp = r"""
      CHECK-LABEL: tfr.func @tf__test_identity_n_op(%x: !tfr.tensor_list) -> (!tfr.tensor_list)

      CHECK-LABEL: tfr.func @tf__test_identity_op(%x: !tfr.tensor) -> (!tfr.tensor) {
      CHECK-NEXT:   %[[list:.*]] = "tfr.build_list"(%x) : (!tfr.tensor) -> !tfr.tensor_list
      CHECK-NEXT:   %[[call:.*]] = tfr.call @tf__test_identity_n_op(%[[list]]) : (!tfr.tensor_list)
    """
    self._check_code(mlir_code, mlir_code_exp)

  def test_quant_builtins(self):
    mlir_code = tfr_gen(sys.modules[__name__], '_tfr_quant', [test_ops])
    mlir_code_exp = r"""
      CHECK-LABEL: tfr.func @tf__test_identity_op(%x: !tfr.tensor) -> (!tfr.tensor) {
      CHECK-NEXT:   %[[raw_data:.*]] = tfr.quant_raw_data(%x) : (!tfr.tensor) -> (!tfr.tensor)
      CHECK-NEXT:   %[[qparam:.*]]:2 = tfr.quant_qparam(%x) : (!tfr.tensor) -> (!tfr.tensor, !tfr.tensor)
      CHECK:        %[[list:.*]] = "tfr.build_list"(%[[qparam]]#0, %[[qparam]]#0) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
      CHECK:        %[[factor:.*]] = tfr.quant_scale_factor(%{{.*}}, %[[list]]) : (f32, !tfr.tensor_list) -> (!tfr.tensor)
      CHECK:        %[[list1:.*]] = "tfr.build_list"(%[[factor]]) : (!tfr.tensor) -> !tfr.tensor_list
      CHECK:        %[[factor1:.*]] = tfr.quant_scale_factor(%{{.*}}, %[[list1]]) : (f32, !tfr.tensor_list) -> (!tfr.tensor)
      CHECK-NEXT:   %[[Sub:.*]] = tfr.call @tf__sub(%[[raw_data]], %[[qparam]]#1) : (!tfr.tensor, !tfr.tensor) -> (!tfr.tensor)
      CHECK:        %[[act_range:.*]]:2 = tfr.quant_act_range(%{{.*}}, %{{.*}}, %{{.*}}) : (!tfr.attr, f32, i64) -> (!tfr.tensor, !tfr.tensor)
      CHECK:        %[[rescale:.*]] = tfr.quant_rescale(%[[Sub]], %[[factor1]], %{{.*}}) : (!tfr.tensor, !tfr.tensor, i64) -> (!tfr.tensor)
      CHECK:        %[[attr:.*]] = tfr.constant i16 -> !tfr.attr
      CHECK:        %[[Cast:.*]] = tfr.call @tf__cast(%[[rescale]], %[[attr]], %{{.*}}) : (!tfr.tensor, !tfr.attr, i1) -> (!tfr.tensor)
      CHECK:        %[[attr_1:.*]] = tfr.constant i8 -> !tfr.attr
      CHECK:        tfr.call @tf__cast(%[[Cast]], %[[attr_1]], %{{.*}}) : (!tfr.tensor, !tfr.attr, i1) -> (!tfr.tensor)
      CHECK:       }

      CHECK-LABEL: tfr.func @tf__test_identity_n_op(%x: !tfr.tensor_list) -> (!tfr.tensor_list) {
      CHECK-NEXT:   %[[raw_data:.*]] = tfr.quant_raw_data(%x) : (!tfr.tensor_list) -> (!tfr.tensor_list)
      CHECK:        tfr.return %[[raw_data:.*]] : !tfr.tensor_list
      CHECK:       }
    """
    self._check_code(mlir_code, mlir_code_exp)


if __name__ == '__main__':
  test.main()
