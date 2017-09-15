# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data import Dataset
from tensorflow.contrib.eager.python import datasets
from tensorflow.python.eager import test
from tensorflow.python.ops import math_ops


class IteratorTest(test.TestCase):

  def testBasic(self):
    got = []
    for t in datasets.Iterator(Dataset.range(4)):
      got.append(t.numpy())
    self.assertAllEqual([0, 1, 2, 3], got)

  def testMultipleIteratorsOnTheSameDataset(self):
    ds = Dataset.range(4)
    it1 = datasets.Iterator(ds)
    it2 = datasets.Iterator(ds)
    got = [x.numpy() for x in it1]
    self.assertAllEqual([0, 1, 2, 3], got)

    got = [x.numpy() for x in it2]
    self.assertAllEqual([0, 1, 2, 3], got)

  def testNestedOutputs(self):
    ds = Dataset.zip((Dataset.range(4), Dataset.zip((Dataset.range(4),
                                                     Dataset.range(4)))))
    total = 0
    # The Iterator will return a nested structure of Tensor objects.
    # Some funkiness to compare against simple integers.
    for (i, x) in enumerate(datasets.Iterator(ds)):
      want = (i, (i, i))
      got = (x[0].numpy(), (x[1][0].numpy(), x[1][1].numpy()))
      self.assertEqual(got, want)
      total += 1
    self.assertEqual(4, total)

  def testMapAndFilter(self):
    def even(x):
      return math_ops.equal(math_ops.mod(x, 2), 0)

    it = datasets.Iterator(Dataset.range(8).map(math_ops.square).filter(even))
    got = [x.numpy() for x in it]
    self.assertAllEqual([0, 4, 16, 36], got)


if __name__ == '__main__':
  test.main()
