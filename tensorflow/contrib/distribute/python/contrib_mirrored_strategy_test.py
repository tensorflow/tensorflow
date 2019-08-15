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
"""Tests the contrib MirroredStrategy specific features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import test

contrib_mirrored_strategies = [
    combinations.NamedDistribution(
        "ContribMirrored1CPU",
        lambda: mirrored_strategy.MirroredStrategy(["/cpu:0"])),
    combinations.NamedDistribution(
        "ContribMirrored1GPU",
        lambda: mirrored_strategy.MirroredStrategy(["/gpu:0"]),
        required_gpus=1),
    combinations.NamedDistribution(
        "ContribMirroredCPUAndGPU",
        lambda: mirrored_strategy.MirroredStrategy(["/cpu:0", "/gpu:0"]),
        required_gpus=1),
    combinations.NamedDistribution(
        "ContribMirrored2GPU",
        lambda: mirrored_strategy.MirroredStrategy(["/gpu:0", "/gpu:1"]),
        required_gpus=2),
]


def all_strategy_and_eager_plus_graph():
  return combinations.times(
      combinations.combine(distribution=contrib_mirrored_strategies),
      combinations.combine(mode=["eager", "graph"]))


class ContribMirroredStrategyTest(test.TestCase, parameterized.TestCase):

  def _initialize_and_evaluate_iterator(self, iterator):
    if context.executing_eagerly():
      iterator.initialize()
      res = iterator.get_next()
      if isinstance(res, values.PerReplica):
        res = res.values
    else:
      with self.cached_session() as sess:
        sess.run(iterator.initialize())
        res = iterator.get_next()
        if isinstance(res, values.PerReplica):
          res = sess.run(res.values)
        else:
          res = sess.run(res)

    return res

  @combinations.generate(all_strategy_and_eager_plus_graph())
  def test_dataset_iterator(self, distribution):
    data = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    dataset = dataset_ops.Dataset.from_tensors(data).repeat()
    iterator = distribution.make_dataset_iterator(dataset)
    res = self._initialize_and_evaluate_iterator(iterator)

    if isinstance(res, tuple):
      self.assertLen(res, 2)
      self.assertAllEqual(data, res[0])
      self.assertAllEqual(data, res[1])
    else:
      self.assertAllEqual(data, res)


if __name__ == "__main__":
  test.main()
