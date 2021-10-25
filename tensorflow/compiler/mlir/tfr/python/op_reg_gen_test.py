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
"""Tests for `op_reg_gen` module."""

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=g-direct-tensorflow-import

import sys

from tensorflow.compiler.mlir.python.mlir_wrapper import filecheck_wrapper as fw
from tensorflow.compiler.mlir.tfr.python import composite
from tensorflow.compiler.mlir.tfr.python.op_reg_gen import gen_register_op
from tensorflow.python.platform import test


Composite = composite.Composite


@composite.Composite(
    'TestNoOp', derived_attrs=['T: numbertype'], outputs=['o1: T'])
def _composite_no_op():
  pass


@Composite(
    'TestCompositeOp',
    inputs=['x: T', 'y: T'],
    attrs=['act: {"", "relu"}', 'trans: bool = true'],
    derived_attrs=['T: numbertype'],
    outputs=['o1: T', 'o2: T'])
def _composite_op(x, y, act, trans):
  return x + act, y + trans


class TFRGenTensorTest(test.TestCase):
  """MLIR Generation Tests for MLIR TFR Program."""

  def test_op_reg_gen(self):
    cxx_code = gen_register_op(sys.modules[__name__])
    cxx_code_exp = r"""
      CHECK: #include "tensorflow/core/framework/op.h"
      CHECK-EMPTY
      CHECK: namespace tensorflow {
      CHECK-EMPTY
      CHECK-LABEL: REGISTER_OP("TestNoOp")
      CHECK-NEXT:      .Attr("T: numbertype")
      CHECK-NEXT:      .Output("o1: T");
      CHECK-EMPTY
      CHECK-LABEL: REGISTER_OP("TestCompositeOp")
      CHECK-NEXT:      .Input("x: T")
      CHECK-NEXT:      .Input("y: T")
      CHECK-NEXT:      .Attr("act: {'', 'relu'}")
      CHECK-NEXT:      .Attr("trans: bool = true")
      CHECK-NEXT:      .Attr("T: numbertype")
      CHECK-NEXT:      .Output("o1: T")
      CHECK-NEXT:      .Output("o2: T");
      CHECK-EMPTY
      CHECK:  }  // namespace tensorflow
    """
    self.assertTrue(fw.check(str(cxx_code), cxx_code_exp), str(cxx_code))


if __name__ == '__main__':
  test.main()
