# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.platform import test
from tensorflow.tensorboard.backend.event_processing import reservoir


class ReservoirTest(test.TestCase):

  def testEmptyReservoir(self):
    r = reservoir.Reservoir(1)
    self.assertFalse(r.Keys())

  def testRespectsSize(self):
    r = reservoir.Reservoir(42)
    self.assertEqual(r._buckets['meaning of life']._max_size, 42)

  def testItemsAndKeys(self):
    r = reservoir.Reservoir(42)
    r.AddItem('foo', 4)
    r.AddItem('bar', 9)
    r.AddItem('foo', 19)
    self.assertItemsEqual(r.Keys(), ['foo', 'bar'])
    self.assertEqual(r.Items('foo'), [4, 19])
    self.assertEqual(r.Items('bar'), [9])

  def testExceptions(self):
    with self.assertRaises(ValueError):
      reservoir.Reservoir(-1)
    with self.assertRaises(ValueError):
      reservoir.Reservoir(13.3)

    r = reservoir.Reservoir(12)
    with self.assertRaises(KeyError):
      r.Items('missing key')

  def testDeterminism(self):
    """Tests that the reservoir is deterministic."""
    key = 'key'
    r1 = reservoir.Reservoir(10)
    r2 = reservoir.Reservoir(10)
    for i in xrange(100):
      r1.AddItem('key', i)
      r2.AddItem('key', i)

    self.assertEqual(r1.Items(key), r2.Items(key))

  def testBucketDeterminism(self):
    """Tests that reservoirs are deterministic at a bucket level.

    This means that only the order elements are added within a bucket matters.
    """
    separate_reservoir = reservoir.Reservoir(10)
    interleaved_reservoir = reservoir.Reservoir(10)
    for i in xrange(100):
      separate_reservoir.AddItem('key1', i)
    for i in xrange(100):
      separate_reservoir.AddItem('key2', i)
    for i in xrange(100):
      interleaved_reservoir.AddItem('key1', i)
      interleaved_reservoir.AddItem('key2', i)

    for key in ['key1', 'key2']:
      self.assertEqual(
          separate_reservoir.Items(key), interleaved_reservoir.Items(key))

  def testUsesSeed(self):
    """Tests that reservoirs with different seeds keep different samples."""
    key = 'key'
    r1 = reservoir.Reservoir(10, seed=0)
    r2 = reservoir.Reservoir(10, seed=1)
    for i in xrange(100):
      r1.AddItem('key', i)
      r2.AddItem('key', i)
    self.assertNotEqual(r1.Items(key), r2.Items(key))

  def testFilterItemsByKey(self):
    r = reservoir.Reservoir(100, seed=0)
    for i in xrange(10):
      r.AddItem('key1', i)
      r.AddItem('key2', i)

    self.assertEqual(len(r.Items('key1')), 10)
    self.assertEqual(len(r.Items('key2')), 10)

    self.assertEqual(r.FilterItems(lambda x: x <= 7, 'key2'), 2)
    self.assertEqual(len(r.Items('key2')), 8)
    self.assertEqual(len(r.Items('key1')), 10)

    self.assertEqual(r.FilterItems(lambda x: x <= 3, 'key1'), 6)
    self.assertEqual(len(r.Items('key1')), 4)
    self.assertEqual(len(r.Items('key2')), 8)


class ReservoirBucketTest(test.TestCase):

  def testEmptyBucket(self):
    b = reservoir._ReservoirBucket(1)
    self.assertFalse(b.Items())

  def testFillToSize(self):
    b = reservoir._ReservoirBucket(100)
    for i in xrange(100):
      b.AddItem(i)
    self.assertEqual(b.Items(), list(xrange(100)))
    self.assertEqual(b._num_items_seen, 100)

  def testDoesntOverfill(self):
    b = reservoir._ReservoirBucket(10)
    for i in xrange(1000):
      b.AddItem(i)
    self.assertEqual(len(b.Items()), 10)
    self.assertEqual(b._num_items_seen, 1000)

  def testMaintainsOrder(self):
    b = reservoir._ReservoirBucket(100)
    for i in xrange(10000):
      b.AddItem(i)
    items = b.Items()
    prev = -1
    for item in items:
      self.assertTrue(item > prev)
      prev = item

  def testKeepsLatestItem(self):
    b = reservoir._ReservoirBucket(5)
    for i in xrange(100):
      b.AddItem(i)
      last = b.Items()[-1]
      self.assertEqual(last, i)

  def testSizeOneBucket(self):
    b = reservoir._ReservoirBucket(1)
    for i in xrange(20):
      b.AddItem(i)
      self.assertEqual(b.Items(), [i])
    self.assertEqual(b._num_items_seen, 20)

  def testSizeZeroBucket(self):
    b = reservoir._ReservoirBucket(0)
    for i in xrange(20):
      b.AddItem(i)
      self.assertEqual(b.Items(), list(range(i + 1)))
    self.assertEqual(b._num_items_seen, 20)

  def testSizeRequirement(self):
    with self.assertRaises(ValueError):
      reservoir._ReservoirBucket(-1)
    with self.assertRaises(ValueError):
      reservoir._ReservoirBucket(10.3)

  def testRemovesItems(self):
    b = reservoir._ReservoirBucket(100)
    for i in xrange(10):
      b.AddItem(i)
    self.assertEqual(len(b.Items()), 10)
    self.assertEqual(b._num_items_seen, 10)
    self.assertEqual(b.FilterItems(lambda x: x <= 7), 2)
    self.assertEqual(len(b.Items()), 8)
    self.assertEqual(b._num_items_seen, 8)

  def testRemovesItemsWhenItemsAreReplaced(self):
    b = reservoir._ReservoirBucket(100)
    for i in xrange(10000):
      b.AddItem(i)
    self.assertEqual(b._num_items_seen, 10000)

    # Remove items
    num_removed = b.FilterItems(lambda x: x <= 7)
    self.assertGreater(num_removed, 92)
    self.assertEqual([], [item for item in b.Items() if item > 7])
    self.assertEqual(b._num_items_seen,
                     int(round(10000 * (1 - float(num_removed) / 100))))

  def testLazyFunctionEvaluationAndAlwaysKeepLast(self):

    class FakeRandom(object):

      def randint(self, a, b):  # pylint:disable=unused-argument
        return 999

    class Incrementer(object):

      def __init__(self):
        self.n = 0

      def increment_and_double(self, x):
        self.n += 1
        return x * 2

    # We've mocked the randomness generator, so that once it is full, the last
    # item will never get durable reservoir inclusion. Since always_keep_last is
    # false, the function should only get invoked 100 times while filling up
    # the reservoir. This laziness property is an essential performance
    # optimization.
    b = reservoir._ReservoirBucket(100, FakeRandom(), always_keep_last=False)
    incrementer = Incrementer()
    for i in xrange(1000):
      b.AddItem(i, incrementer.increment_and_double)
    self.assertEqual(incrementer.n, 100)
    self.assertEqual(b.Items(), [x * 2 for x in xrange(100)])

    # This time, we will always keep the last item, meaning that the function
    # should get invoked once for every item we add.
    b = reservoir._ReservoirBucket(100, FakeRandom(), always_keep_last=True)
    incrementer = Incrementer()

    for i in xrange(1000):
      b.AddItem(i, incrementer.increment_and_double)
    self.assertEqual(incrementer.n, 1000)
    self.assertEqual(b.Items(), [x * 2 for x in xrange(99)] + [999 * 2])


class ReservoirBucketStatisticalDistributionTest(test.TestCase):

  def setUp(self):
    self.total = 1000000
    self.samples = 10000
    self.n_buckets = 100
    self.total_per_bucket = self.total // self.n_buckets
    self.assertEqual(self.total % self.n_buckets, 0, 'total must be evenly '
                     'divisible by the number of buckets')
    self.assertTrue(self.total > self.samples, 'need to have more items '
                    'than samples')

  def AssertBinomialQuantity(self, measured):
    p = 1.0 * self.n_buckets / self.samples
    mean = p * self.samples
    variance = p * (1 - p) * self.samples
    error = measured - mean
    # Given that the buckets were actually binomially distributed, this
    # fails with probability ~2E-9
    passed = error * error <= 36.0 * variance
    self.assertTrue(passed, 'found a bucket with measured %d '
                    'too far from expected %d' % (measured, mean))

  def testBucketReservoirSamplingViaStatisticalProperties(self):
    # Not related to a 'ReservoirBucket', but instead number of buckets we put
    # samples into for testing the shape of the distribution
    b = reservoir._ReservoirBucket(_max_size=self.samples)
    # add one extra item because we always keep the most recent item, which
    # would skew the distribution; we can just slice it off the end instead.
    for i in xrange(self.total + 1):
      b.AddItem(i)

    divbins = [0] * self.n_buckets
    modbins = [0] * self.n_buckets
    # Slice off the last item when we iterate.
    for item in b.Items()[0:-1]:
      divbins[item // self.total_per_bucket] += 1
      modbins[item % self.n_buckets] += 1

    for bucket_index in xrange(self.n_buckets):
      divbin = divbins[bucket_index]
      modbin = modbins[bucket_index]
      self.AssertBinomialQuantity(divbin)
      self.AssertBinomialQuantity(modbin)


if __name__ == '__main__':
  test.main()
