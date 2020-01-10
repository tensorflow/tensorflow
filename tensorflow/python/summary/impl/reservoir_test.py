import tensorflow.python.platform

from tensorflow.python.platform import googletest
from tensorflow.python.summary.impl import reservoir


class ReservoirTest(googletest.TestCase):

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
      self.assertEqual(separate_reservoir.Items(key),
                       interleaved_reservoir.Items(key))

  def testUsesSeed(self):
    """Tests that reservoirs with different seeds keep different samples."""
    key = 'key'
    r1 = reservoir.Reservoir(10, seed=0)
    r2 = reservoir.Reservoir(10, seed=1)
    for i in xrange(100):
      r1.AddItem('key', i)
      r2.AddItem('key', i)
    self.assertNotEqual(r1.Items(key), r2.Items(key))


class ReservoirBucketTest(googletest.TestCase):

  def testEmptyBucket(self):
    b = reservoir._ReservoirBucket(1)
    self.assertFalse(b.Items())

  def testFillToSize(self):
    b = reservoir._ReservoirBucket(100)
    for i in xrange(100):
      b.AddItem(i)
    self.assertEqual(b.Items(), range(100))

  def testDoesntOverfill(self):
    b = reservoir._ReservoirBucket(10)
    for i in xrange(1000):
      b.AddItem(i)
    self.assertEqual(len(b.Items()), 10)

  def testMaintainsOrder(self):
    b = reservoir._ReservoirBucket(100)
    for i in xrange(10000):
      b.AddItem(i)
    items = b.Items()
    prev = None
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

  def testSizeZeroBucket(self):
    b = reservoir._ReservoirBucket(0)
    for i in xrange(20):
      b.AddItem(i)
      self.assertEqual(b.Items(), range(i+1))

  def testSizeRequirement(self):
    with self.assertRaises(ValueError):
      reservoir._ReservoirBucket(-1)
    with self.assertRaises(ValueError):
      reservoir._ReservoirBucket(10.3)


class ReservoirBucketStatisticalDistributionTest(googletest.TestCase):

  def setUp(self):
    self.total = 1000000
    self.samples = 10000
    self.n_buckets = 100
    self.total_per_bucket = self.total / self.n_buckets
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
      divbins[item / self.total_per_bucket] += 1
      modbins[item % self.n_buckets] += 1

    for bucket_index in xrange(self.n_buckets):
      divbin = divbins[bucket_index]
      modbin = modbins[bucket_index]
      self.AssertBinomialQuantity(divbin)
      self.AssertBinomialQuantity(modbin)


if __name__ == '__main__':
  googletest.main()
