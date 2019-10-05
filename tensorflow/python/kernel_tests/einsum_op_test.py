# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.Einsum."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


class EinsumOpTest(test.TestCase):

  def _check(self, s, *input_shapes, **kwargs):
    dtype = kwargs.pop('dtype', np.float32)
    r = np.random.RandomState(0)
    inputs = []
    for shape in input_shapes:
      arr = np.array(r.randn(*shape)).astype(dtype)
      if dtype == np.complex64 or dtype == np.complex128:
        arr += 1j * np.array(r.randn(*shape)).astype(dtype)
      inputs.append(arr)
    input_tensors = [constant_op.constant(x, shape=x.shape) for x in inputs]
    a = np.einsum(s, *inputs)
    b = self.evaluate(gen_linalg_ops.einsum(input_tensors, s))
    self.assertAllClose(a, b, atol=1e-4, rtol=1e-4)

  def testUnary(self):
    self._check('->', ())
    self._check('ab->', (3, 3))
    self._check('ab->ab', (3, 3))
    self._check('abc->b', (3, 4, 5))
    self._check('abc->ca', (3, 4, 5))
    self._check('abc->cab', (3, 4, 5))

  def testUnaryWithRepeatedLabels(self):
    self._check('aa->', (3, 3))
    self._check('aa->a', (3, 3))
    self._check('aaa->', (3, 3, 3))
    self._check('aaa->a', (3, 3, 3))
    self._check('aab->a', (3, 3, 4))
    self._check('aabcc->a', (3, 3, 5, 4, 4))
    self._check('aabcc->ac', (3, 3, 5, 4, 4))
    self._check('aabcd->ad', (3, 3, 5, 4, 4))

  def testUnaryEllipsis(self):
    # Unary cases with ellipsis.
    # Edge cases.
    self._check('...->...', ())
    self._check('...->', ())
    self._check('->...', ())

    # Tests from dask
    self._check('a...a->a...', (2, 2))
    self._check('a...a->', (2, 2))
    self._check('a...a->...', (2, 5, 1, 2))
    self._check('a...a->a...', (2, 1, 2))
    self._check('a...a->a...', (2, 3, 4, 5, 2))

    # Regular cases.
    self._check('...ijk->...ki', (3, 4, 5))
    self._check('...ijk->...ki', (1, 3, 4, 5))
    self._check('...ijk->...ki', (2, 2, 3, 4, 5))

    # Repeated indices.
    self._check('i...ii->...i', (3, 2, 3, 3))

  def testBinarySimple(self):
    # Binary cases in XLA mode must have either (a) each index appearing exactly
    # once in both the inputs (batch or contraction index), or (b) appearing
    # exactly once in an input and in the output (free index).
    self._check(',->', (), ())
    self._check('a,a->', (3,), (3,))
    self._check('a,a->a', (3,), (3,))
    self._check('ab,b->a', (3, 4), (4,))
    self._check('ab,ab->', (3, 4), (3, 4))
    self._check('ab,bc->ac', (3, 4), (4, 5))
    self._check('nij,jk->nik', (5, 2, 3), (3, 4))
    self._check('abc,bad->abcd', (1, 2, 3), (2, 1, 4))
    # Based on https://github.com/google/jax/issues/37#issuecomment-448572187
    self._check('sa,shb->shab', (2, 1), (2, 3, 4))

  def testReducedIndices(self):
    self._check('ba,b->', (3, 2), (3,))
    self._check('ab,ab->', (3, 4), (3, 4))
    self._check('abce,badf->abcd', (1, 2, 3, 4), (2, 1, 4, 3))

  def testRepeatedIndices(self):
    # Repeated indices.
    self._check('ijj,k->ik', (2, 3, 3), (4,))
    self._check('aba,a->b', (3, 4, 3), (3,))
    # From https://github.com/dask/dask/pull/3412#discussion_r182413444
    self._check('aab,bc->ac', (2, 2, 3), (3, 4))
    self._check('aab,bcc->ac', (2, 2, 3), (3, 4, 4))

  def testEllipsis(self):
    # Batch matmul with ellipsis but without broadcasting.
    self._check('...mk,...kn->...mn', (5, 1, 2, 3), (5, 1, 3, 4))
    # Empty batch dimensions.
    self._check('...mk,...kn->...mn', (2, 3), (3, 4))
    # Tensor contraction with transpose.
    self._check('...ija,aijb...->ba...ij', (1, 2, 2, 3, 1), (1, 2, 3, 4, 1, 2))
    # Output subscripts may omit ellipsis when batch shape is empty.
    self._check('...mk,...kn->mn', (2, 3), (3, 4))
    self._check('...mk,kn->mn', (2, 3), (3, 4))
    self._check('mk,...kn->mn', (2, 3), (3, 4))

  def testBroadcasting(self):
    # Batch matmul with broadcasting.
    self._check('...ij,...jk->...ik', (1, 2, 3), (3, 5))
    self._check('...ij,...jk->...ik', (2, 3), (1, 3, 5))
    self._check('...ij,...jk->...ik', (5, 2, 3), (3, 5))
    self._check('...ij,...jk->...ik', (2, 3), (5, 3, 5))
    self._check('...ij,...jk->...ik', (3, 1, 2, 3), (1, 1, 7, 3, 5))
    self._check('i...j,j...k->...ik', (2, 1, 3, 1, 3), (3, 1, 7, 5))
    # Following 2 from https://stackoverflow.com/a/19203475/1611416
    self._check('...abc,...abcd->...d', (1, 1, 2, 3, 4), (5, 2, 3, 4, 6))
    self._check('ab...,b->ab...', (2, 3, 1, 1, 5), (3,))
    self._check('i...j,j...k->i...k', (3, 1, 2, 2), (2, 2, 3, 1, 4))

  def testBroadcastingWithRepeatedIndices(self):
    # Broadcasting with repeated indices.
    self._check('ij,jk...k->i...', (3, 2), (2, 4, 1, 4))
    self._check('ij,jk...k->...i', (3, 2), (2, 4, 5, 4))
    self._check('ijj,jk...k->i...', (3, 2, 2), (2, 4, 1, 4))
    self._check('i...jj,jk...k->i...', (3, 3, 1, 2, 2), (2, 4, 1, 5, 4))

  def testDtypes(self):
    bfloat16 = dtypes.bfloat16.as_numpy_dtype

    def check(dtype):
      r = np.random.RandomState(0)
      equation = 'ij,jk->ik'
      input_shapes = [(2, 2), (2, 2)]
      inputs = []
      for shape in input_shapes:
        arr = np.array(r.randn(*shape)).astype(dtype)
        if dtype == np.complex64 or dtype == np.complex128:
          arr += 1j * np.array(r.randn(*shape)).astype(dtype)
        inputs.append(arr)
      input_tensors = [constant_op.constant(x) for x in inputs]
      if dtype == bfloat16:
        # np.einsum doesn't support bfloat16.
        a = np.einsum(equation,
                      *[x.astype(np.float32) for x in inputs]).astype(dtype)
      else:
        a = np.einsum(equation, *inputs)

      b = self.evaluate(gen_linalg_ops.einsum(input_tensors, equation))
      tol = 1e-2 if dtype == bfloat16 else 1e-4
      self.assertAllClose(a, b, atol=tol, rtol=tol)

    for dtype in [
        bfloat16, np.float32, np.float64, np.complex64, np.complex128, np.int32,
        np.int64
    ]:
      check(dtype)

  @test_util.disable_xla('b/131919749')
  @test_util.run_in_graph_and_eager_modes
  def testInvalid(self):
    r = np.random.RandomState(0)
    cases = [
        # incorrect rank.
        ('ij,jk->ik', r.randn(1, 2, 3), r.randn(3, 4)),
        ('...ij,jk->ik', r.randn(3), r.randn(3, 4)),
        # inconsistent dimensions.
        ('ij,jk->ik', r.randn(2, 3), r.randn(4, 4)),
        # broadcasting is invalid
        ('...ij,...jk->...ik', r.randn(5, 2, 3), r.randn(7, 3, 4)),
        # output should have ellipsis when broadcasting shape is
        # non-empty.
        ('...ij,...jk->ik', r.randn(2, 2, 3), r.randn(3, 4)),
    ]
    for args in cases:
      with self.assertRaises((ValueError, errors.InvalidArgumentError)):
        _ = self.evaluate(gen_linalg_ops.einsum(args[1:], args[0]))

      placeholders = [
          array_ops.placeholder_with_default(x, shape=None) for x in args[1:]
      ]
      with self.assertRaises((ValueError, errors.InvalidArgumentError)):
        _ = self.evaluate(gen_linalg_ops.einsum(placeholders, args[0]))

  @test_util.run_in_graph_and_eager_modes
  def testPlaceholder(self):

    def check(equation, *input_and_placeholder_shapes):
      r = np.random.RandomState(0)
      inputs = []
      input_placeholders = []
      for actual_shape, placeholder_shape in input_and_placeholder_shapes:
        input_np = np.array(r.randn(*actual_shape))
        inputs.append(input_np)
        input_placeholders.append(
            array_ops.placeholder_with_default(input_np, placeholder_shape))

      a = np.einsum(equation, *inputs)
      b = self.evaluate(gen_linalg_ops.einsum(input_placeholders, equation))
      self.assertAllClose(a, b, atol=1e-4, rtol=1e-4)

    check('bijl,bjkm->bik', ((9, 2, 3, 5), (None, None, None, 5)),
          ((9, 3, 4, 7), (None, None, 4, None)))
    check('bijl,bjkm->bik', ((9, 2, 3, 5), None), ((9, 3, 4, 7), None))
    check('...ij,...->...i', ((4, 3, 1, 2), (None, 3, None, 2)),
          ((4, 3), (None, 3)))
    check('...ij,...jk->...ik', ((3, 1, 2, 3), None), ((1, 7, 3, 4), None))

  @test_util.disable_xla('b/131919749')
  def testOutputRepeatedLabels(self):
    # This is the reverse operation of generalized traces, to be used for
    # computing symbolic gradients of einsum. Note: this operation is not
    # supported by np.einsum as it's only required for gradients.
    r = np.random.RandomState(0)
    a = r.randn(2, 2)
    s = 'a->aa'
    diag_a = np.diag(np.diag(a))
    b = self.evaluate(gen_linalg_ops.einsum([np.diag(a)], s))
    self.assertAllClose(diag_a, b, atol=1e-4, rtol=1e-4)

  def testEmpty(self):
    def check(equation, input_shapes, output_shape):
      # All these cases result in an output filled with zeros, so we don't call
      # np.einsum. Also np.einsum doesn't support generalized diagonals which
      # are needed for EinsumOp gradients.
      r = np.random.RandomState(0)
      inputs = [np.array(r.randn(*shape)) for shape in input_shapes]
      output = self.evaluate(gen_linalg_ops.einsum(inputs, equation))
      self.assertAllClose(output, np.zeros(output_shape), atol=1e-4, rtol=1e-4)

    # Contractions along zero-sized dimensons.
    check('ab,bc->ac', [(0, 10), (10, 10)], (0, 10))
    # From transformer xl.
    check('ibnd,ijbn->jnd', [(1, 0, 5, 10), (1, 1, 0, 5)], (1, 5, 10))

  @test_util.disable_xla('b/131919749')
  def testEmptyWithRepeatedLabels(self):

    def check(equation, input_shapes, output_shape):
      # All these cases result in an output filled with zeros, so we don't call
      # np.einsum. Also np.einsum doesn't support generalized diagonals which
      # are needed for EinsumOp gradients.
      r = np.random.RandomState(0)
      inputs = [np.array(r.randn(*shape)) for shape in input_shapes]
      output = self.evaluate(gen_linalg_ops.einsum(inputs, equation))
      self.assertAllClose(output, np.zeros(output_shape), atol=1e-4, rtol=1e-4)

    # Generalized traces with zero-sized dimensions.
    check('aab,bc->ac', [(0, 0, 10), (10, 10)], (0, 10))
    check('aaab,bc->c', [(0, 0, 0, 3), (3, 4)], (4,))
    # Generalized diagonals along with contraction.
    check('ab,bc->aaca', [(0, 10), (10, 5)], (0, 0, 5, 0))
    check('ab,bc->aaa', [(0, 10), (10, 5)], (0, 0, 0))
    check('ab,bc->cc', [(0, 10), (10, 5)], (5, 5))
    check('ab,ab->aaa', [(0, 5), (0, 5)], (0, 0, 0))


@test_util.run_all_in_graph_and_eager_modes
class EinsumGradTest(test.TestCase):

  def _check_gradient(self, s, *input_shapes):
    with self.cached_session():
      r = np.random.RandomState(0)
      inputs = [np.array(r.randn(*shape), np.float64) for shape in input_shapes]
      input_tensors = [constant_op.constant(x, shape=x.shape) for x in inputs]
      analytical, numerical = gradient_checker_v2.compute_gradient(
          lambda *xs: gen_linalg_ops.einsum(xs, s), input_tensors)
      self.assertLess(
          gradient_checker_v2.max_error(analytical, numerical), 1e-4)

  @test_util.disable_xla('b/131919749')
  def testUnary(self):
    # Unary cases.
    self._check_gradient('->', ())
    self._check_gradient('aaa->a', (3, 3, 3))
    self._check_gradient('aabcd->ad', (3, 3, 5, 4, 4))
    self._check_gradient('aabcd->add', (3, 3, 5, 4, 4))
    self._check_gradient('abcd->da', (3, 5, 4, 2))

  @test_util.disable_xla('b/131919749')
  def testUnaryEllipsis(self):
    self._check_gradient('...->...', ())
    self._check_gradient('...->', ())
    self._check_gradient('->...', ())

    # Tests from dask
    self._check_gradient('a...a->a...', (2, 2))
    self._check_gradient('a...a->', (2, 2))
    self._check_gradient('a...a->...', (2, 5, 1, 2))
    self._check_gradient('a...a->a...', (2, 1, 2))
    self._check_gradient('a...a->a...', (2, 3, 4, 5, 2))

    self._check_gradient('...ijk->...ki', (3, 4, 5))
    self._check_gradient('...ijk->...ki', (1, 3, 4, 5))
    self._check_gradient('...ijk->...ki', (2, 2, 3, 4, 5))
    self._check_gradient('ab...cd->da...', (3, 5, 2, 3, 4, 2))

  def testBinarySimple(self):
    # Binary cases in XLA mode must have either (a) each index appearing exactly
    # once in both the inputs (batch or contraction index), or (b) appearing
    # exactly once in an input and in the output (free index).
    self._check_gradient(',->', (), ())
    self._check_gradient('a,a->', (3,), (3,))
    self._check_gradient('a,a->a', (3,), (3,))
    self._check_gradient('ab,b->a', (3, 4), (4,))
    self._check_gradient('ab,ab->', (3, 4), (3, 4))
    self._check_gradient('ab,bc->ac', (3, 4), (4, 5))
    self._check_gradient('nij,jk->nik', (5, 2, 3), (3, 4))
    self._check_gradient('abc,bad->abcd', (1, 2, 3), (2, 1, 4))
    # Based on https://github.com/google/jax/issues/37#issuecomment-448572187
    self._check_gradient('sa,shb->shab', (2, 1), (2, 3, 4))

  def testEmpty(self):
    # From Transformer XL.
    self._check_gradient('ibnd,ijbn->jnd', (1, 0, 5, 10), (1, 1, 0, 5))

  def testReducedIndices(self):
    self._check_gradient('ba,b->', (3, 2), (3,))
    self._check_gradient('ab,ab->', (3, 4), (3, 4))
    self._check_gradient('ijkm,ijln->ijmn', (2, 3, 3, 4), (2, 3, 3, 2))
    self._check_gradient('abce,badf->abcd', (1, 2, 3, 4), (2, 1, 4, 3))

  @test_util.disable_xla('b/131919749')
  def testReducedIndicesWithRepeatedLabels(self):
    self._check_gradient('abce,badf->bcba', (1, 2, 3, 4), (2, 1, 4, 3))

  @test_util.disable_xla('b/131919749')
  def testRepeatedLabels(self):
    # Repeated indices.
    self._check_gradient('aba,a->b', (3, 4, 3), (3,))
    self._check_gradient('ijj,k->ik', (2, 3, 3), (4,))
    self._check_gradient('ill,k->ik', (2, 3, 3), (4,))
    # From https://github.com/dask/dask/pull/3412#discussion_r182413444
    self._check_gradient('aab,bc->ac', (1, 1, 3), (3, 4))
    self._check_gradient('aab,bcc->ac', (2, 2, 3), (3, 4, 4))

  @test_util.disable_xla('b/131919749')
  def testEmptyWithRepeatedLabels(self):
    self._check_gradient('aab,bc->ac', (0, 0, 10), (10, 10))
    self._check_gradient('aab,bc->ac', (1, 1, 0), (0, 10))
    self._check_gradient('aaab,bc->c', (0, 0, 0, 3), (3, 4))

  def testBroadcasting(self):
    self._check_gradient('...ij,...jk->...ik', (3, 2), (2, 4))
    self._check_gradient('ij...,jk...->ik...', (3, 2, 1), (2, 4))
    self._check_gradient('...ij,...jk->...ik', (3, 1, 3, 2), (1, 5, 2, 4))
    self._check_gradient('i...j,j...k->i...k', (3, 1, 2, 2), (2, 2, 3, 1, 4))

  @test_util.disable_xla('b/131919749')
  def testBroadcastingWithRepeatedLabels(self):
    self._check_gradient('ij,jk...k->i...', (3, 2), (2, 4, 1, 4))
    self._check_gradient('aab,b...c->a...c', (1, 1, 3), (3, 1, 1, 4))


class EinsumBenchmark(test.Benchmark):
  cases = [
      # Unary cases.
      ['ijk->i', 100],
      ['ijk->kji', 100],
      # Regular matmul or batch matmul.
      ['ij,jk->ik', 1000],
      ['ji,kj->ik', 1000],
      ['ab,ab->', 100],
      ['ab,ba->', 100],
      ['abc,abc->', 100],
      ['abc,bac->', 100],
      ['abc,cba->', 100],
      ['bij,bjk->bik', 100],
      ['bji,bjk->bki', 100],
      ['ikl,kji->kl', 100],
      ['klj,lki->ij', 100],
      ['ijk,ilj->kli', 100],
      ['kij,mkb->ijmb', 100],
      ['abcd,ad->bc', 40],
      # Larger binary contractions.
      ['ijk,jklm->il', 40],
      ['efabc,eabcd->efd', 30],
      ['fabec,abcde->fde', 30],
      ['efabc,edabc->efd', 30],
      ['eadbf,dfebc->ecfad', 30],
      ['abcdef,bcdfg->abcdeg', 30],
  ]

  def benchmarkEinsum(self):
    for equation, dim in self.cases:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device('/cpu:0'):
        r = np.random.RandomState(0)
        input_subscripts = equation.split('->')[0].split(',')
        input_vars = []
        for subscript in input_subscripts:
          input_shape = (dim,) * len(subscript)
          input_vars.append(
              variables.Variable(np.array(r.randn(*input_shape), np.float32)))
        variables.global_variables_initializer().run()

        # Call einsum_v1.
        self.run_op_benchmark(
            sess,
            special_math_ops.einsum(equation, *input_vars),
            min_iters=50,
            name='einsum_v1_cpu_({})_{}'.format(equation, dim))

        # Call gen_linalg_ops.einsum.
        self.run_op_benchmark(
            sess,
            gen_linalg_ops.einsum(input_vars, equation),
            min_iters=50,
            name='einsum_v2_cpu_({})_{}'.format(equation, dim))


if __name__ == '__main__':
  test.main()
