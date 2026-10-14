# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.nn.flash_attention and the FlashAttention ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.platform import test


def _reference(query, key, value, scale, is_causal, keep_mask=None, rate=0.0):
  """Unfused float32 attention on [batch, seq, heads, dim] inputs."""
  query, key, value = (
      math_ops.cast(t, dtypes.float32) for t in (query, key, value)
  )
  logits = special_math_ops.einsum("bqhd,bkhd->bhqk", query, key) * scale
  if is_causal:
    seq = array_ops.shape(logits)[-1]
    allowed = math_ops.range(seq)[:, None] >= math_ops.range(seq)[None, :]
    logits = array_ops.where_v2(allowed, logits, -1e9)
  probs = nn_ops.softmax(logits, axis=-1)
  if keep_mask is not None:
    probs = probs * keep_mask / (1.0 - rate)
  return special_math_ops.einsum("bhqk,bkhd->bqhd", probs, value)


def _value_and_grads(fn, query, key, value, output_grad):
  with backprop.GradientTape() as tape:
    tape.watch([query, key, value])
    output = fn(query, key, value)
    loss = math_ops.reduce_sum(math_ops.cast(output, dtypes.float32) *
                               output_grad)
  return [output] + tape.gradient(loss, [query, key, value])


def _tolerance(dtype):
  return {dtypes.float32: 1e-4, dtypes.float16: 2e-2, dtypes.bfloat16: 5e-2}[
      dtype
  ]


def _sm80_gpu_available():
  return test_util.is_gpu_available(min_cuda_compute_capability=(8, 0))


class FlashAttentionTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # The float32 reference must not use TensorFloat-32 matmuls on Ampere+.
    config.enable_tensor_float_32_execution(False)
    self.addCleanup(config.enable_tensor_float_32_execution, True)

  def _random(self, shape, dtype, seed):
    # Round to `dtype` so the float32 reference sees exactly the same inputs.
    x = random_ops.random_normal(shape, seed=seed)
    return math_ops.cast(math_ops.cast(x, dtype), dtypes.float32)

  def _check(self, dtype, batch, seq_q, seq_kv, heads, head_dim,
             head_dim_v=None, is_causal=False, jit_compile=False):
    head_dim_v = head_dim_v or head_dim
    query = self._random([batch, seq_q, heads, head_dim], dtype, 1)
    key = self._random([batch, seq_kv, heads, head_dim], dtype, 2)
    value = self._random([batch, seq_kv, heads, head_dim_v], dtype, 3)
    output_grad = random_ops.random_normal(
        [batch, seq_q, heads, head_dim_v], seed=4)
    scale = 1.0 / np.sqrt(head_dim)

    def flash(q, k, v):
      return nn_ops.flash_attention(q, k, v, is_causal=is_causal)

    def run_flash(q, k, v):
      return _value_and_grads(flash, q, k, v, output_grad)

    if jit_compile:
      run_flash = def_function.function(run_flash, jit_compile=True)
    got = run_flash(*(math_ops.cast(t, dtype) for t in (query, key, value)))
    want = _value_and_grads(
        lambda q, k, v: _reference(q, k, v, scale, is_causal),
        query, key, value, output_grad)

    self.assertEqual(got[0].dtype, dtype)
    self.assertEqual(got[0].shape, [batch, seq_q, heads, head_dim_v])
    tol = _tolerance(dtype)
    for name, g, w in zip(["output", "dq", "dk", "dv"], got, want):
      g = self.evaluate(math_ops.cast(g, dtypes.float32))
      w = self.evaluate(w)
      # Relative to the largest magnitude, since some gradients are ~0.
      err = np.max(np.abs(g - w)) / max(np.max(np.abs(w)), 1e-3)
      self.assertFalse(np.isnan(g).any(), msg=name)
      self.assertLess(err, tol, msg=name)

  @parameterized.product(
      dtype=[dtypes.float32, dtypes.float16, dtypes.bfloat16],
      head_dim=[16, 64, 128],
      is_causal=[False, True],
  )
  def testMatchesReferenceCPU(self, dtype, head_dim, is_causal):
    with test_util.force_cpu():
      self._check(dtype, batch=2, seq_q=67, seq_kv=67, heads=3,
                  head_dim=head_dim, is_causal=is_causal)

  @parameterized.product(
      dtype=[dtypes.float32, dtypes.float16, dtypes.bfloat16],
      head_dim=[16, 64, 128],
      is_causal=[False, True],
  )
  def testMatchesReferenceGPU(self, dtype, head_dim, is_causal):
    if not test_util.is_gpu_available():
      self.skipTest("Covered by testMatchesReferenceCPU without a GPU")
    self._check(dtype, batch=2, seq_q=67, seq_kv=67, heads=3,
                head_dim=head_dim, is_causal=is_causal)

  @parameterized.parameters(dtypes.float16, dtypes.bfloat16)
  def testCpuGradientsLongSequence(self, dtype):
    # dK and dV sum over every query; the CPU kernel must accumulate them in
    # float32, not in `dtype`, or the error grows with the sequence length.
    with test_util.force_cpu():
      query = self._random([1, 1024, 1, 32], dtype, 1)
      key = self._random([1, 1024, 1, 32], dtype, 2)
      value = self._random([1, 1024, 1, 32], dtype, 3)
      output_grad = random_ops.random_normal([1, 1024, 1, 32], seed=4)
      got = _value_and_grads(
          nn_ops.flash_attention,
          *(math_ops.cast(t, dtype) for t in (query, key, value)), output_grad)
      want = _value_and_grads(
          lambda q, k, v: _reference(q, k, v, 1.0 / np.sqrt(32), False),
          query, key, value, output_grad)
      for name, g, w in zip(["output", "dq", "dk", "dv"], got, want):
        g = self.evaluate(math_ops.cast(g, dtypes.float32))
        w = self.evaluate(w)
        err = np.max(np.abs(g - w)) / max(np.max(np.abs(w)), 1e-3)
        self.assertLess(err, 2e-2, msg=name)

  @parameterized.parameters(dtypes.float32, dtypes.float16)
  def testDifferentSequenceLengthsAndValueDim(self, dtype):
    self._check(dtype, batch=2, seq_q=33, seq_kv=101, heads=2, head_dim=64,
                head_dim_v=32)

  def testSingleToken(self):
    self._check(dtypes.float32, batch=1, seq_q=1, seq_kv=1, heads=1,
                head_dim=32, is_causal=True)

  @parameterized.product(
      dtype=[dtypes.float16, dtypes.bfloat16],
      head_dim=[32, 40, 96, 128, 160, 192, 256],
      is_causal=[False, True],
  )
  def testFlashAttention2(self, dtype, head_dim, is_causal):
    if not _sm80_gpu_available():
      self.skipTest("FlashAttention-2 needs a GPU with compute capability 8.0+")
    self._check(dtype, batch=2, seq_q=257, seq_kv=257, heads=2,
                head_dim=head_dim, is_causal=is_causal)

  @parameterized.product(
      dtype=[dtypes.float32, dtypes.float16, dtypes.bfloat16],
      is_causal=[False, True],
  )
  def testXla(self, dtype, is_causal):
    if not test_util.is_gpu_available():
      self.skipTest("FlashAttention is only compiled by XLA on GPU")
    if dtype != dtypes.float32 and not _sm80_gpu_available():
      self.skipTest("Needs a GPU with compute capability 8.0+")
    self._check(dtype, batch=2, seq_q=100, seq_kv=100, heads=2, head_dim=64,
                is_causal=is_causal, jit_compile=True)

  def testDropoutIsDeterministicGivenSeed(self):
    query = random_ops.random_normal([2, 64, 2, 32], seed=1)
    run = lambda seed: self.evaluate(nn_ops.flash_attention(
        query, query, query, dropout_rate=0.3, seed=seed))
    self.assertAllEqual(run([1, 2]), run([1, 2]))
    self.assertNotAllClose(run([1, 2]), run([1, 3]))
    self.assertAllEqual(run(7), run([7, 0]))

  @parameterized.product(
      dtype=[dtypes.float32, dtypes.float16, dtypes.bfloat16],
      is_causal=[False, True],
  )
  def testDropout(self, dtype, is_causal):
    if dtype != dtypes.float32 and not test_util.is_gpu_available():
      self.skipTest("Only float32 is checked on CPU")
    rate, batch, heads, seq = 0.2, 2, 2, 64
    seed = [5, 6]
    query = self._random([batch, seq, heads, seq], dtype, 1)
    key = self._random([batch, seq, heads, seq], dtype, 2)
    value = self._random([batch, seq, heads, seq], dtype, 3)
    output_grad = random_ops.random_normal([batch, seq, heads, seq], seed=4)
    scale = 1.0 / np.sqrt(seq)

    def flash(q, k, v):
      return nn_ops.flash_attention(
          q, k, v, is_causal=is_causal, dropout_rate=rate, seed=seed)

    # The output is linear in value, so value = identity (seq_kv == head_dim)
    # returns the dropped and rescaled attention weights, which reveals the
    # op's dropout mask: [batch, seq_q, heads, seq_kv] -> [b, h, q, k].
    eye = array_ops.broadcast_to(
        linalg_ops.eye(seq, dtype=dtype)[None, :, None, :],
        [batch, seq, heads, seq])
    weights = flash(math_ops.cast(query, dtype), math_ops.cast(key, dtype), eye)
    keep_mask = math_ops.cast(
        array_ops.transpose(weights, [0, 2, 1, 3]) > 0, dtypes.float32)

    valid = array_ops.ones_like(keep_mask)
    if is_causal:
      valid = array_ops.matrix_band_part(valid, -1, 0)
    drop_fraction = 1.0 - self.evaluate(
        math_ops.reduce_sum(keep_mask) / math_ops.reduce_sum(valid))
    self.assertAllClose(drop_fraction, rate, atol=0.03)

    got = _value_and_grads(
        flash, *(math_ops.cast(t, dtype) for t in (query, key, value)),
        output_grad)
    want = _value_and_grads(
        lambda q, k, v: _reference(q, k, v, scale, is_causal, keep_mask, rate),
        query, key, value, output_grad)
    tol = _tolerance(dtype)
    for name, g, w in zip(["output", "dq", "dk", "dv"], got, want):
      g = self.evaluate(math_ops.cast(g, dtypes.float32))
      w = self.evaluate(w)
      err = np.max(np.abs(g - w)) / max(np.max(np.abs(w)), 1e-3)
      self.assertLess(err, tol, msg=name)

  def testStaticShape(self):
    query = array_ops.zeros([2, 5, 3, 8])
    value = array_ops.zeros([2, 7, 3, 4])
    key = array_ops.zeros([2, 7, 3, 8])
    self.assertEqual(
        nn_ops.flash_attention(query, key, value).shape, [2, 5, 3, 4])

  def testInvalidArguments(self):
    x = array_ops.zeros([2, 4, 1, 8])
    with self.assertRaisesRegex(ValueError, "must be rank 4"):
      nn_ops.flash_attention(x[0], x[0], x[0])
    for rate in (-0.1, 1.0):
      with self.assertRaisesRegex(errors.InvalidArgumentError, "dropout_rate"):
        self.evaluate(nn_ops.flash_attention(x, x, x, dropout_rate=rate))
    # Shape mismatches are caught statically by the Python wrapper.
    with self.assertRaisesRegex(ValueError, "must be rank 4"):
      nn_ops.flash_attention(x, array_ops.zeros([2, 4, 16]), x)
    with self.assertRaisesRegex(ValueError, "incompatible"):  # num_heads
      nn_ops.flash_attention(x, array_ops.zeros([2, 4, 2, 8]), x)
    with self.assertRaisesRegex(ValueError, "incompatible"):  # head_dim
      nn_ops.flash_attention(x, array_ops.zeros([2, 4, 1, 32]), x)
    # The kernel checks them too, for callers of the raw op.
    seed = constant_op.constant(0, dtypes.uint64)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "feature size mismatch"):
      self.evaluate(gen_nn_ops.flash_attention(
          x[:, :, 0], array_ops.zeros([2, 4, 32]), x[:, :, 0], seed, seed))
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError), "is_causal"):
      self.evaluate(nn_ops.flash_attention(
          x, array_ops.zeros([2, 5, 1, 8]), array_ops.zeros([2, 5, 1, 8]),
          is_causal=True))

  @parameterized.parameters(dtypes.float32, dtypes.float16)
  def testEmptySequences(self, dtype):
    # Empty inputs skip the kernels; the outputs must still be zero.
    def run(seq_q, seq_kv):
      query = array_ops.ones([2, seq_q, 1, 32], dtype)
      kv = array_ops.ones([2, seq_kv, 1, 32], dtype)
      output_grad = array_ops.ones([2, seq_q, 1, 32])
      return [self.evaluate(t) for t in _value_and_grads(
          nn_ops.flash_attention, query, kv, kv, output_grad)]

    output, dq, dk, dv = run(seq_q=8, seq_kv=0)
    self.assertAllEqual(output, np.zeros([2, 8, 1, 32]))
    self.assertAllEqual(dq, np.zeros([2, 8, 1, 32]))
    self.assertEqual(dk.shape, (2, 0, 1, 32))
    output, dq, dk, dv = run(seq_q=0, seq_kv=8)
    self.assertEqual(output.shape, (2, 0, 1, 32))
    self.assertAllEqual(dk, np.zeros([2, 8, 1, 32]))
    self.assertAllEqual(dv, np.zeros([2, 8, 1, 32]))

    # test batch size of zero
    query_b0 = array_ops.ones([0, 8, 1, 32], dtype)
    kv_b0 = array_ops.ones([0, 8, 1, 32], dtype)
    og_b0 = array_ops.ones([0, 8, 1, 32])
    out_b0, dq_b0, dk_b0, dv_b0 = [self.evaluate(t) for t in _value_and_grads(
        nn_ops.flash_attention, query_b0, kv_b0, kv_b0, og_b0)]
    self.assertEqual(out_b0.shape, (0, 8, 1, 32))
    self.assertEqual(dq_b0.shape, (0, 8, 1, 32))
    self.assertEqual(dk_b0.shape, (0, 8, 1, 32))
    self.assertEqual(dv_b0.shape, (0, 8, 1, 32))

  def testXlaInvalidDropout(self):
    if not test_util.is_gpu_available():
      self.skipTest("FlashAttention is only compiled by XLA on GPU")
    x = array_ops.zeros([2, 4, 1, 8])
    fn = def_function.function(
        lambda: nn_ops.flash_attention(x, x, x, dropout_rate=1.0),
        jit_compile=True)
    with self.assertRaisesRegex(errors.InvalidArgumentError, "dropout_rate"):
      self.evaluate(fn())

  def testDimsExceedingInt32(self):
    # Zero-sized, so a dimension over the int32 limit doesn't need memory.
    seed = constant_op.constant(0, dtypes.uint64)
    query = array_ops.zeros([1, 1, 0])
    key = array_ops.zeros([1, 2**31, 0])
    with self.assertRaisesRegex(errors.InvalidArgumentError, "int32 limit"):
      self.evaluate(gen_nn_ops.flash_attention(query, key, key, seed, seed))

  def testUnsupportedHeadDim(self):
    # float32 always uses the fallback kernels, which stop at 128.
    x = array_ops.zeros([1, 16, 1, 256])
    with self.assertRaisesRegex(errors.InvalidArgumentError, "not supported"):
      self.evaluate(nn_ops.flash_attention(x, x, x))

  def testRawOps(self):
    query = random_ops.random_normal([3, 10, 16], seed=1)
    seed = constant_op.constant(0, dtypes.uint64)
    output, softmax_lse = gen_nn_ops.flash_attention(
        query, query, query, seed, seed, scale=0.25)
    logits = math_ops.matmul(query, query, transpose_b=True) * 0.25
    self.assertAllClose(softmax_lse, math_ops.reduce_logsumexp(logits, -1),
                        atol=1e-4)
    self.assertAllClose(
        output, math_ops.matmul(nn_ops.softmax(logits), query), atol=1e-4)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
