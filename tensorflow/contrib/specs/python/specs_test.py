# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Testing specs specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.specs import python
from tensorflow.contrib.specs.python import summaries
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.math_ops  # pylint: disable=unused-import
from tensorflow.python.platform import test

specs = python


def _rand(*size):
  return np.random.uniform(size=size).astype("f")


class SpecsTest(test.TestCase):

  def testSimpleConv(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 18, 19, 5))
      spec = "net = Cr(64, [5, 5])"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [1, 18, 19, 64])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 18, 19, 64))
      self.assertEqual(
          summaries.tf_spec_structure(spec, inputs),
          "_ variablev2 conv variablev2 biasadd relu")

  def testUnary(self):
    # This is just a quick and dirty check that these ops exist
    # and work as unary ops.
    with self.test_session():
      inputs = constant_op.constant(_rand(17, 55))
      spec = "net = Do(0.5) | Bn | Unit(1) | Relu | Sig | Tanh | Smax"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [17, 55])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (17, 55))

  def testAdd(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(17, 55))
      spec = "net = Fs(10) + Fr(10)"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [17, 10])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (17, 10))
      self.assertEqual(
          summaries.tf_spec_structure(spec, inputs),
          "_ variablev2 dot variablev2 biasadd sig "
          "<> variablev2 dot variablev2 biasadd relu add")

  def testMpPower(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 64, 64, 5))
      spec = "M2 = Mp([2, 2]); net = M2**3"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [1, 8, 8, 5])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 8, 8, 5))
      self.assertEqual(
          summaries.tf_spec_structure(spec, inputs),
          "_ maxpool maxpool maxpool")

  def testAbbrevPower(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 64, 64, 5))
      spec = "C3 = Cr([3, 3]); M2 = Mp([2, 2]); net = (C3(5) | M2)**3"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [1, 8, 8, 5])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 8, 8, 5))
      self.assertEqual(
          summaries.tf_spec_structure(spec, inputs),
          "_ variablev2 conv variablev2 biasadd relu maxpool"
          " variablev2 conv variablev2"
          " biasadd relu maxpool variablev2 conv variablev2"
          " biasadd relu maxpool")

  def testAbbrevPower2(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 64, 64, 5))
      spec = "C3 = Cr(_1=[3, 3]); M2 = Mp([2, 2]);"
      spec += "net = (C3(_0=5) | M2)**3"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [1, 8, 8, 5])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 8, 8, 5))
      self.assertEqual(
          summaries.tf_spec_structure(spec, inputs),
          "_ variablev2 conv variablev2 biasadd relu maxpool"
          " variablev2 conv variablev2 biasadd relu"
          " maxpool variablev2 conv variablev2 biasadd relu"
          " maxpool")

  def testConc(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(10, 20))
      spec = "net = Conc(1, Fs(20), Fs(10))"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [10, 30])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (10, 30))
      self.assertEqual(
          summaries.tf_spec_structure(spec, inputs),
          "_ variablev2 dot variablev2 biasadd sig "
          "<> variablev2 dot variablev2 biasadd sig _ concatv2")

  def testImport(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(10, 20))
      spec = ("S = Import('from tensorflow.python.ops" +
              " import math_ops; f = math_ops.sigmoid')")
      spec += "; net = S | S"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [10, 20])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (10, 20))
      self.assertEqual(summaries.tf_spec_structure(spec, inputs), "_ sig sig")

  def testLstm2(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 64, 64, 5))
      spec = "net = Lstm2(15)"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [1, 64, 64, 15])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 64, 64, 15))

  def testLstm2to1(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 64, 64, 5))
      spec = "net = Lstm2to1(15)"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [1, 64, 15])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 64, 15))

  def testLstm2to0(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(1, 64, 64, 5))
      spec = "net = Lstm2to0(15)"
      outputs = specs.create_net(spec, inputs)
      self.assertEqual(outputs.get_shape().as_list(), [1, 15])
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (1, 15))

  def testKeywordRestriction(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(10, 20))
      spec = "import re; net = Conc(1, Fs(20), Fs(10))"
      self.assertRaises(ValueError, lambda: specs.create_net(spec, inputs))

  def testParams(self):
    params = "x = 3; y = Ui(-10, 10); z = Lf(1, 100); q = Nt(0.0, 1.0)"
    bindings = specs.eval_params(params, {})
    self.assertTrue("x" in bindings)
    self.assertEqual(bindings["x"], 3)
    self.assertTrue("y" in bindings)
    self.assertTrue("z" in bindings)
    self.assertTrue("q" in bindings)

  # XXX: the cleverness of this code is over 9000
  # TODO: original author please fix
  def DISABLED_testSpecsOps(self):
    # pylint: disable=undefined-variable
    with self.assertRaises(NameError):
      _ = Cr
    with specs.ops:
      self.assertIsNotNone(Cr)
      self.assertTrue(callable(Cr(64, [3, 3])))
    with self.assertRaises(NameError):
      _ = Cr

  # XXX: the cleverness of this code is over 9000
  # TODO: original author please fix
  def DISABLED_testVar(self):
    with self.test_session() as sess:
      with specs.ops:
        # pylint: disable=undefined-variable
        v = Var("test_var",
                shape=[2, 2],
                initializer=init_ops.constant_initializer(42.0))
      inputs = constant_op.constant(_rand(10, 100))
      outputs = v.funcall(inputs)
      self.assertEqual(len(variables.global_variables()), 1)
      sess.run([outputs.initializer])
      outputs_value = outputs.eval()
      self.assertEqual(outputs_value.shape, (2, 2))
      self.assertEqual(outputs_value[1, 1], 42.0)

  # XXX: the cleverness of this code is over 9000
  # TODO: original author please fix
  def DISABLED_testShared(self):
    with self.test_session():
      with specs.ops:
        # pylint: disable=undefined-variable
        f = Shared(Fr(100))
        g = f | f | f | f
      inputs = constant_op.constant(_rand(10, 100))
      _ = g.funcall(inputs)
      self.assertEqual(len(variables.global_variables()), 2)


if __name__ == "__main__":
  test.main()
