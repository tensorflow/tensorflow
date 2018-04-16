# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Recurrent ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.recurrent.python.ops import recurrent
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test as test_lib
from tensorflow.python.platform import tf_logging as logging


_ElmanState = collections.namedtuple('ElmanState', ('h'))
_ElmanTheta = collections.namedtuple('ElmanTheta', ('w', 'b'))
_ElmanInputs = collections.namedtuple('ElmanInputs', ('x'))


# TODO(drpng): add test for max length computation.
class RecurrentTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    # pylint:disable=invalid-name
    _PolyState = collections.namedtuple('PolyState', ('value', 'x_power'))
    _PolyTheta = collections.namedtuple('PolyTheta', ('x'))
    _PolyInputs = collections.namedtuple('PolyInputs', ('coeff'))
    # pylint:enable=invalid-name

    def Poly(theta, state, inputs):
      next_state = _PolyState(
          value=state.value + inputs.coeff * state.x_power,
          x_power=state.x_power * theta.x)
      return next_state, []

    with self.test_session() as sess:
      theta = _PolyTheta(x=array_ops.constant(2.0))
      state = _PolyState(
          value=array_ops.constant(0.0),
          x_power=array_ops.constant(1.0))
      inputs = _PolyInputs(coeff=array_ops.constant([1., 2., 3.]))

      # x = 2
      # 1 + 2*x + 3*x^2
      ret = recurrent.Recurrent(theta, state, inputs, Poly)

      acc, state = sess.run(ret)
      self.assertAllClose(acc.value, [1., 5., 17.])
      self.assertAllClose(acc.x_power, [2., 4., 8.])
      self.assertAllClose(state.value, 17.)
      self.assertAllClose(state.x_power, 8.)

      y = ret[1].value
      dx, d_coeff = gradients_impl.gradients(ys=[y], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = sess.run([dx, d_coeff])

      # 2 + 6*x
      self.assertAllClose(dx_val, 14.)
      self.assertAllClose(d_coeff_val, [1., 2., 4.])

      # acc = [1, 1+2x, 1+2x+3x^2]
      # sum(acc) = 3 + 4x + 3x^2
      acc = ret[0].value
      dx, d_coeff = gradients_impl.gradients(
          ys=[math_ops.reduce_sum(acc)], xs=[theta.x, inputs.coeff])
      dx_val, d_coeff_val = sess.run([dx, d_coeff])
      # 4 + 6*x
      self.assertAllClose(dx_val, 16.)
      self.assertAllClose(d_coeff_val, [3., 4., 4.])

  @staticmethod
  def Rand(shape):
    return random_ops.random_uniform(
        shape, minval=-0.2, maxval=0.2, dtype=dtypes.float64)

  @staticmethod
  def Elman(theta, state0, inputs):
    h0, w, b, x = state0.h, theta.w, theta.b, inputs.x
    xw = math_ops.matmul(array_ops.concat([x, h0], axis=1), w)
    h1 = math_ops.sigmoid(xw + b)
    state1 = _ElmanState(h=h1)
    return (state1, state1)

  @staticmethod
  def ElmanGrad(theta, state0, inputs, extras, dstate1):

    @function.Defun()
    def Grad(h0, w, b, x, h1, dh1):
      del b
      # We hand-roll the gradient for the 2nd half of the cell as a demo.
      dxwb = (dh1 * (1 - h1) * h1)
      dxw, db = dxwb, math_ops.reduce_sum(dxwb, axis=0)

      # Uses tf.gradient for the 1nd half of the cell as a demo.
      xw = math_ops.matmul(array_ops.concat([x, h0], axis=1), w)
      dh0, dx, dw = gradients_impl.gradients(
          ys=[xw], xs=[h0, x, w], grad_ys=[dxw])

      return dh0, dx, dw, db

    dh0, dx, dw, db = Grad(state0.h, theta.w, theta.b, inputs.x,
                           extras.h, dstate1.h)
    dstate0 = _ElmanState(h=dh0)
    dinputs = _ElmanInputs(x=dx)
    return (_ElmanTheta(w=dw, b=db), dstate0, dinputs)

  @staticmethod
  def ElmanOut(state1):
    return _ElmanState(x=state1.h)

  @staticmethod
  def ElmanOutGrad(dout):
    return _ElmanState(h=dout.x)

  def testElman(self):
    for seqlen, use_grad in [(1, False), (1, True), (7, False), (7, True)]:
      logging.info('== Elman: seqlen=%s, use_grad=%s', seqlen, use_grad)
      self._ParameterizedTestElman(seqlen, use_grad)

  def _ParameterizedTestElman(self, seqlen, use_grad):

    with self.test_session() as sess:
      random_seed.set_random_seed(342462)

      batch = 3
      dims = 4
      theta = _ElmanTheta(w=RecurrentTest.Rand([2 * dims, dims]),
                          b=RecurrentTest.Rand([dims]))
      state0 = _ElmanState(h=RecurrentTest.Rand([batch, dims]))
      inputs = _ElmanInputs(x=RecurrentTest.Rand([seqlen, batch, dims]))

      # Statically unrolled.
      s = state0
      out = []
      for i in xrange(seqlen):
        inp = _ElmanInputs(x=inputs.x[i, :])
        s, _ = RecurrentTest.Elman(theta, s, inp)
        out += [s.h]
      acc0, final0 = array_ops.stack(out), s.h
      loss0 = math_ops.reduce_sum(acc0) + math_ops.reduce_sum(final0)
      (dw0, db0, dh0, di0) = gradients_impl.gradients(
          loss0, [theta.w, theta.b, state0.h, inputs.x])

      acc1, final1 = recurrent.Recurrent(
          theta=theta,
          state0=state0,
          inputs=inputs,
          cell_fn=RecurrentTest.Elman,
          cell_grad=RecurrentTest.ElmanGrad if use_grad else None)
      assert isinstance(acc1, _ElmanState)
      assert isinstance(final1, _ElmanState)
      acc1, final1 = acc1.h, final1.h
      loss1 = math_ops.reduce_sum(acc1) + math_ops.reduce_sum(final1)
      (dw1, db1, dh1, di1) = gradients_impl.gradients(
          loss1, [theta.w, theta.b, state0.h, inputs.x])

      # Fetches a few values and compare them.
      (acc0, acc1, final0, final1, dw0, dw1, db0, db1, dh0, dh1, di0,
       di1) = sess.run(
           [acc0, acc1, final0, final1, dw0, dw1, db0, db1, dh0, dh1, di0, di1])
      self.assertAllClose(acc0, acc1)
      self.assertAllClose(final0, final1)
      self.assertAllClose(dw0, dw1)
      self.assertAllClose(db0, db1)
      self.assertAllClose(dh0, dh1)
      self.assertAllClose(di0, di1)

if __name__ == '__main__':
  test_lib.main()
