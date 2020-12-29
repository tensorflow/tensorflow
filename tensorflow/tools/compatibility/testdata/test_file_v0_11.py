# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf upgrader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib


class TestUpgrade(test_util.TensorFlowTestCase):
  """Test various APIs that have been changed in 1.0.

  This test will not run in current TensorFlow, but did run in 0.11.
  This file is intended to be converted by a genrule() that uses the converter
  so that a 1.0 compatible version of this file is generated. That is run as
  a unit test if the converter is successful.
  """

  @test_util.run_v1_only("b/120545219")
  def testArgRenames(self):
    with self.cached_session():

      a = [[1., 2., 3.], [4., 5., 6.]]
      b = [[True, False, False], [False, True, True]]
      dim0 = [1]
      dim1 = [1]

      self.assertAllEqual(
          tf.reduce_any(
              b, reduction_indices=dim0).eval(), [True, True])
      self.assertAllEqual(
          tf.reduce_all(
              b, reduction_indices=[0]).eval(), [False, False, False])
      self.assertAllEqual(
          tf.reduce_all(
              b, reduction_indices=dim1).eval(), [False, False])
      self.assertAllEqual(
          tf.reduce_sum(
              a, reduction_indices=[1]).eval(), [6., 15.])
      self.assertAllEqual(
          tf.reduce_sum(
              a, reduction_indices=[0, 1]).eval(), 21.0)
      self.assertAllEqual(tf.reduce_sum(a, [0, 1]).eval(), 21.0)
      self.assertAllEqual(
          tf.reduce_prod(
              a, reduction_indices=[1]).eval(), [6., 120.])
      self.assertAllEqual(
          tf.reduce_prod(
              a, reduction_indices=[0, 1]).eval(), 720.0)
      self.assertAllEqual(tf.reduce_prod(a, [0, 1]).eval(), 720.0)
      self.assertAllEqual(
          tf.reduce_mean(
              a, reduction_indices=[1]).eval(), [2., 5.])
      self.assertAllEqual(
          tf.reduce_mean(
              a, reduction_indices=[0, 1]).eval(), 3.5)
      self.assertAllEqual(tf.reduce_mean(a, [0, 1]).eval(), 3.5)
      self.assertAllEqual(
          tf.reduce_min(
              a, reduction_indices=[1]).eval(), [1., 4.])
      self.assertAllEqual(
          tf.reduce_min(
              a, reduction_indices=[0, 1]).eval(), 1.0)
      self.assertAllEqual(tf.reduce_min(a, [0, 1]).eval(), 1.0)
      self.assertAllEqual(
          tf.reduce_max(
              a, reduction_indices=[1]).eval(), [3., 6.])
      self.assertAllEqual(
          tf.reduce_max(
              a, reduction_indices=[0, 1]).eval(), 6.0)
      self.assertAllEqual(tf.reduce_max(a, [0, 1]).eval(), 6.0)
      self.assertAllClose(tf.reduce_logsumexp(a, reduction_indices=[1]).eval(),
                          [3.40760589, 6.40760612])
      self.assertAllClose(
          tf.reduce_logsumexp(a, reduction_indices=[0, 1]).eval(),
          6.45619344711)
      self.assertAllClose(
          tf.reduce_logsumexp(a, [0, 1]).eval(), 6.45619344711)
      self.assertAllEqual(
          tf.expand_dims([[1, 2], [3, 4]], axis=1).eval(),
          [[[1, 2]], [[3, 4]]])

  @test_util.run_v1_only("b/120545219")
  def testArgMinMax(self):
    with self.cached_session():
      self.assertAllEqual(
          tf.argmin([[1, 2, 3], [4, 1, 0]], dimension=1).eval(),
          [0, 2])
      self.assertAllEqual(
          tf.argmin([[1, 2, 3], [4, 1, 0]], dimension=0).eval(),
          [0, 1, 1])
      self.assertAllEqual(
          tf.argmax([[1, 2, 3], [4, 1, 0]], dimension=1).eval(),
          [2, 0])
      self.assertAllEqual(
          tf.argmax([[1, 2, 3], [4, 1, 0]], dimension=0).eval(),
          [1, 0, 0])

  @test_util.run_v1_only("b/120545219")
  def testExpandAndSqueeze(self):
    with self.cached_session():

      # TODO(aselle): sparse_split, sparse_reduce_sum,
      #  sparse_reduce_sum_sparse, reduce_join
      a = [[1, 2, 3]]
      self.assertAllEqual(tf.expand_dims(tf.squeeze(a, [0]), 0).eval(),
                          a)
      self.assertAllEqual(tf.squeeze(tf.expand_dims(a, 1), [1]).eval(),
                          a)
      self.assertAllEqual(
          tf.expand_dims(tf.squeeze([[1, 2, 3]], axis=[0]), dim=0).eval(), a)
      self.assertAllEqual(
          tf.squeeze(tf.expand_dims([[1, 2, 3]], dim=1), axis=[1]).eval(), a)

      self.assertAllEqual(
          tf.squeeze(tf.expand_dims([[1, 2, 3]], dim=1), axis=[1]).eval(), a)

  @test_util.run_v1_only("b/120545219")
  def testArithmeticRenames(self):
    with self.cached_session() as s:
      stuff = tf.split(1, 2, [[1, 2, 3, 4], [4, 5, 6, 7]])
      vals = s.run(stuff)
      self.assertAllEqual(vals,
                          [[[1, 2], [4, 5]], [[3, 4], [6, 7]]])
      self.assertAllEqual(
          tf.neg(tf.mul(tf.add(1, 2), tf.sub(5, 3))).eval(),
          -6)
      self.assertAllEqual(
          s.run(tf.listdiff([1, 2, 3], [3, 3, 4]))[0], [1, 2])
      self.assertAllEqual(
          tf.list_diff([1, 2, 3], [3, 3, 4])[0].eval(), [1, 2])
      a = [[1., 2., 3.], [4., 5., 6.]]
      foo = np.where(np.less(a, 2), np.negative(a), a)
      self.assertAllEqual(
          tf.select(tf.less(a, 2), tf.neg(a), a).eval(),
          foo)
      self.assertAllEqual(
          tf.complex_abs(tf.constant(3 + 4.j)).eval(),
          5)
      #     # TODO(aselle): (tf.batch_*)
      # ]

  @test_util.run_v1_only("b/120545219")
  def testBatchAndSvd(self):
    with self.cached_session():
      mat = [[1., 2.], [2., 3.]]
      batched_mat = tf.expand_dims(mat, [0])
      result = tf.matmul(mat, mat).eval()
      result_batched = tf.batch_matmul(batched_mat, batched_mat).eval()
      self.assertAllEqual(result_batched, np.expand_dims(result, 0))
      self.assertAllEqual(
          tf.svd(mat, False, True).eval(),
          tf.svd(mat, compute_uv=False, full_matrices=True).eval())

  @test_util.run_v1_only("b/120545219")
  def testCrossEntropy(self):
    # TODO(aselle): Test sparse_softmax_...
    with self.cached_session():
      labels = [.8, .5, .2, .1]
      logits = [.9, .1, .3, .1]
      self.assertAllEqual(
          tf.nn.softmax_cross_entropy_with_logits(
              logits, labels).eval(),
          tf.nn.softmax_cross_entropy_with_logits(
              labels=labels, logits=logits).eval())
      self.assertAllEqual(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits, labels).eval(),
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=labels, logits=logits).eval())

  @test_util.run_v1_only("b/120545219")
  def testVariables(self):
    with self.cached_session() as s:

      # make some variables
      _ = [tf.Variable([1, 2, 3], dtype=tf.float32),
           tf.Variable([1, 2, 3], dtype=tf.int32)]
      s.run(tf.global_variables_initializer())
      _ = [v.name for v in tf.all_variables()]
      _ = [v.name for v in tf.local_variables()]

  @test_util.run_v1_only("b/120545219")
  def testSummaries(self):
    with self.cached_session() as s:
      var = tf.Variable([1, 2, 3], dtype=tf.float32)
      s.run(tf.global_variables_initializer())
      x, y = np.meshgrid(np.linspace(-10, 10, 256), np.linspace(-10, 10, 256))
      image = np.sin(x**2 + y**2) / np.sqrt(x**2 + y**2) * .5 + .5
      image = image[None, :, :, None]

      # make a dummy sound
      freq = 440  # A = 440Hz
      sampling_frequency = 11000
      audio = np.sin(2 * np.pi * np.linspace(0, 1, sampling_frequency) * freq)
      audio = audio[None, :, None]
      test_dir = tempfile.mkdtemp()
      # test summaries
      writer = tf.train.SummaryWriter(test_dir)
      summaries = [
          tf.scalar_summary("scalar_var", var[0]),
          tf.scalar_summary("scalar_reduce_var", tf.reduce_sum(var)),
          tf.histogram_summary("var_histogram", var),
          tf.image_summary("sin_image", image),
          tf.audio_summary("sin_wave", audio, sampling_frequency),
      ]
      run_summaries = s.run(summaries)
      writer.add_summary(s.run(tf.merge_summary(inputs=run_summaries)))
      # This is redundant, but we want to be able to rewrite the command
      writer.add_summary(s.run(tf.merge_all_summaries()))
      writer.close()
      shutil.rmtree(test_dir)


if __name__ == "__main__":
  test_lib.main()
