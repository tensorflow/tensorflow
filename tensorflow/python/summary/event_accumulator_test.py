import os

import tensorflow.python.platform

import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.summary import event_accumulator as ea


class _EventGenerator(object):

  def __init__(self):
    self.items = []

  def Load(self):
    while self.items:
      yield self.items.pop(0)

  def AddScalar(self, tag, wall_time=0, step=0, value=0):
    event = tf.Event(
        wall_time=wall_time, step=step,
        summary=tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)]
        )
    )
    self.AddEvent(event)

  def AddHistogram(self, tag, wall_time=0, step=0, hmin=1, hmax=2, hnum=3,
                   hsum=4, hsum_squares=5, hbucket_limit=None, hbucket=None):
    histo = tf.HistogramProto(min=hmin, max=hmax, num=hnum, sum=hsum,
                              sum_squares=hsum_squares,
                              bucket_limit=hbucket_limit,
                              bucket=hbucket)
    event = tf.Event(
        wall_time=wall_time,
        step=step,
        summary=tf.Summary(value=[tf.Summary.Value(tag=tag, histo=histo)]))
    self.AddEvent(event)

  def AddImage(self, tag, wall_time=0, step=0, encoded_image_string='imgstr',
               width=150, height=100):
    image = tf.Summary.Image(encoded_image_string=encoded_image_string,
                             width=width, height=height)
    event = tf.Event(
        wall_time=wall_time,
        step=step,
        summary=tf.Summary(
            value=[tf.Summary.Value(tag=tag, image=image)]))
    self.AddEvent(event)

  def AddEvent(self, event):
    self.items.append(event)


class EventAccumulatorTest(tf.test.TestCase):

  def assertTagsEqual(self, tags1, tags2):
    # Make sure the two dictionaries have the same keys.
    self.assertItemsEqual(tags1, tags2)
    # Additionally, make sure each key in the dictionary maps to the same value.
    for key in tags1:
      if isinstance(tags1[key], list):
        # We don't care about the order of the values in lists, thus asserting
        # only if the items are equal.
        self.assertItemsEqual(tags1[key], tags2[key])
      else:
        # Make sure the values are equal.
        self.assertEqual(tags1[key], tags2[key])


class MockingEventAccumulatorTest(EventAccumulatorTest):

  def setUp(self):
    super(MockingEventAccumulatorTest, self).setUp()
    self.empty = {ea.IMAGES: [],
                  ea.SCALARS: [],
                  ea.HISTOGRAMS: [],
                  ea.COMPRESSED_HISTOGRAMS: [],
                  ea.GRAPH: False}
    self._real_constructor = ea.EventAccumulator
    self._real_generator = ea._GeneratorFromPath
    def _FakeAccumulatorConstructor(generator, *args, **kwargs):
      ea._GeneratorFromPath = lambda x: generator
      return self._real_constructor(generator, *args, **kwargs)
    ea.EventAccumulator = _FakeAccumulatorConstructor

  def tearDown(self):
    ea.EventAccumulator = self._real_constructor
    ea._GeneratorFromPath = self._real_generator

  def testEmptyAccumulator(self):
    gen = _EventGenerator()
    x = ea.EventAccumulator(gen)
    x.Reload()
    self.assertEqual(x.Tags(), self.empty)

  def testTags(self):
    gen = _EventGenerator()
    gen.AddScalar('sv1')
    gen.AddScalar('sv2')
    gen.AddHistogram('hst1')
    gen.AddHistogram('hst2')
    gen.AddImage('im1')
    gen.AddImage('im2')
    acc = ea.EventAccumulator(gen)
    acc.Reload()
    self.assertTagsEqual(
        acc.Tags(), {
            ea.IMAGES: ['im1', 'im2'],
            ea.SCALARS: ['sv1', 'sv2'],
            ea.HISTOGRAMS: ['hst1', 'hst2'],
            ea.COMPRESSED_HISTOGRAMS: ['hst1', 'hst2'],
            ea.GRAPH: False})

  def testReload(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    acc.Reload()
    self.assertEqual(acc.Tags(), self.empty)
    gen.AddScalar('sv1')
    gen.AddScalar('sv2')
    gen.AddHistogram('hst1')
    gen.AddHistogram('hst2')
    gen.AddImage('im1')
    gen.AddImage('im2')
    self.assertEqual(acc.Tags(), self.empty)
    acc.Reload()
    self.assertTagsEqual(acc.Tags(), {
        ea.IMAGES: ['im1', 'im2'],
        ea.SCALARS: ['sv1', 'sv2'],
        ea.HISTOGRAMS: ['hst1', 'hst2'],
        ea.COMPRESSED_HISTOGRAMS: ['hst1', 'hst2'],
        ea.GRAPH: False})

  def testScalars(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    sv1 = ea.ScalarEvent(wall_time=1, step=10, value=32)
    sv2 = ea.ScalarEvent(wall_time=2, step=12, value=64)
    gen.AddScalar('sv1', wall_time=1, step=10, value=32)
    gen.AddScalar('sv2', wall_time=2, step=12, value=64)
    acc.Reload()
    self.assertEqual(acc.Scalars('sv1'), [sv1])
    self.assertEqual(acc.Scalars('sv2'), [sv2])

  def testHistograms(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)

    val1 = ea.HistogramValue(min=1, max=2, num=3, sum=4, sum_squares=5,
                             bucket_limit=[1, 2, 3], bucket=[0, 3, 0])
    val2 = ea.HistogramValue(min=-2, max=3, num=4, sum=5, sum_squares=6,
                             bucket_limit=[2, 3, 4], bucket=[1, 3, 0])

    hst1 = ea.HistogramEvent(wall_time=1, step=10, histogram_value=val1)
    hst2 = ea.HistogramEvent(wall_time=2, step=12, histogram_value=val2)
    gen.AddHistogram('hst1', wall_time=1, step=10, hmin=1, hmax=2, hnum=3,
                     hsum=4, hsum_squares=5, hbucket_limit=[1, 2, 3],
                     hbucket=[0, 3, 0])
    gen.AddHistogram('hst2', wall_time=2, step=12, hmin=-2, hmax=3, hnum=4,
                     hsum=5, hsum_squares=6, hbucket_limit=[2, 3, 4],
                     hbucket=[1, 3, 0])
    acc.Reload()
    self.assertEqual(acc.Histograms('hst1'), [hst1])
    self.assertEqual(acc.Histograms('hst2'), [hst2])

  def testCompressedHistograms(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen, compression_bps=(0, 2500, 5000, 7500, 10000))

    gen.AddHistogram('hst1', wall_time=1, step=10, hmin=1, hmax=2, hnum=3,
                     hsum=4, hsum_squares=5, hbucket_limit=[1, 2, 3],
                     hbucket=[0, 3, 0])
    gen.AddHistogram('hst2', wall_time=2, step=12, hmin=-2, hmax=3, hnum=4,
                     hsum=5, hsum_squares=6, hbucket_limit=[2, 3, 4],
                     hbucket=[1, 3, 0])
    acc.Reload()

    # Create the expected values after compressing hst1
    expected_vals1 = [ea.CompressedHistogramValue(bp, val) for bp, val in [(
        0, 1.0), (2500, 1.25), (5000, 1.5), (7500, 1.75), (10000, 2.0)]]
    expected_cmphst1 = ea.CompressedHistogramEvent(
        wall_time=1,
        step=10,
        compressed_histogram_values=expected_vals1)
    self.assertEqual(acc.CompressedHistograms('hst1'), [expected_cmphst1])

    # Create the expected values after compressing hst2
    expected_vals2 = [
        ea.CompressedHistogramValue(bp, val)
        for bp, val in [(0, -2), (2500, 2), (5000, 2 + float(1) / 3), (
            7500, 2 + float(2) / 3), (10000, 3)]
    ]
    expected_cmphst2 = ea.CompressedHistogramEvent(
        wall_time=2,
        step=12,
        compressed_histogram_values=expected_vals2)
    self.assertEqual(acc.CompressedHistograms('hst2'), [expected_cmphst2])

  def testPercentile(self):

    def AssertExpectedForBps(bps, expected):
      output = acc._Percentile(
          bps, bucket_limit, cumsum_weights, histo_min, histo_max, histo_num)
      self.assertAlmostEqual(expected, output)

    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)

    bucket_limit = [1, 2, 3, 4]
    histo_num = 100

    ## All weights in the first bucket
    cumsum_weights = [10000, 10000, 10000, 10000]
    histo_min = -1
    histo_max = .9
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, acc._Remap(2500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(5000, acc._Remap(5000, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(7500, acc._Remap(7500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## All weights in second bucket
    cumsum_weights = [0, 10000, 10000, 10000]
    histo_min = 1.1
    histo_max = 1.8
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, acc._Remap(2500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(5000, acc._Remap(5000, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(7500, acc._Remap(7500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## All weights in the last bucket
    cumsum_weights = [0, 0, 0, 10000]
    histo_min = 3.1
    histo_max = 3.6
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, acc._Remap(2500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(5000, acc._Remap(5000, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(7500, acc._Remap(7500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## Weights distributed between two buckets
    cumsum_weights = [0, 4000, 10000, 10000]
    histo_min = 1.1
    histo_max = 2.9
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, acc._Remap(2500, 0, 4000, histo_min,
                                          bucket_limit[1]))
    AssertExpectedForBps(5000, acc._Remap(5000, 4000, 10000, bucket_limit[1],
                                          histo_max))
    AssertExpectedForBps(7500, acc._Remap(7500, 4000, 10000, bucket_limit[1],
                                          histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## Weights distributed between all buckets
    cumsum_weights = [1000, 4000, 8000, 10000]
    histo_min = -1
    histo_max = 3.9
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, acc._Remap(2500, 1000, 4000, bucket_limit[0],
                                          bucket_limit[1]))
    AssertExpectedForBps(5000, acc._Remap(5000, 4000, 8000, bucket_limit[1],
                                          bucket_limit[2]))
    AssertExpectedForBps(7500, acc._Remap(7500, 4000, 8000, bucket_limit[1],
                                          bucket_limit[2]))
    AssertExpectedForBps(9000, acc._Remap(9000, 8000, 10000, bucket_limit[2],
                                          histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## Most weight in first bucket
    cumsum_weights = [9000, 10000, 10000, 10000]
    histo_min = -1
    histo_max = 1.1
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, acc._Remap(2500, 0, 9000, histo_min,
                                          bucket_limit[0]))
    AssertExpectedForBps(5000, acc._Remap(5000, 0, 9000, histo_min,
                                          bucket_limit[0]))
    AssertExpectedForBps(7500, acc._Remap(7500, 0, 9000, histo_min,
                                          bucket_limit[0]))
    AssertExpectedForBps(9500, acc._Remap(9500, 9000, 10000, bucket_limit[0],
                                          histo_max))
    AssertExpectedForBps(10000, histo_max)

  def testImages(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    im1 = ea.ImageEvent(wall_time=1, step=10, encoded_image_string='big',
                        width=400, height=300)
    im2 = ea.ImageEvent(wall_time=2, step=12, encoded_image_string='small',
                        width=40, height=30)
    gen.AddImage('im1', wall_time=1, step=10, encoded_image_string='big',
                 width=400, height=300)
    gen.AddImage('im2', wall_time=2, step=12, encoded_image_string='small',
                 width=40, height=30)
    acc.Reload()
    self.assertEqual(acc.Images('im1'), [im1])
    self.assertEqual(acc.Images('im2'), [im2])

  def testActivation(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    self.assertFalse(acc._activated)
    with self.assertRaises(RuntimeError):
      acc.Tags()
    with self.assertRaises(RuntimeError):
      acc.Scalars('sv1')
    acc.Reload()
    self.assertTrue(acc._activated)
    acc._activated = False

  def testKeyError(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    acc.Reload()
    with self.assertRaises(KeyError):
      acc.Scalars('sv1')
    with self.assertRaises(KeyError):
      acc.Scalars('hst1')
    with self.assertRaises(KeyError):
      acc.Scalars('im1')
    with self.assertRaises(KeyError):
      acc.Histograms('sv1')
    with self.assertRaises(KeyError):
      acc.Histograms('im1')
    with self.assertRaises(KeyError):
      acc.Images('sv1')
    with self.assertRaises(KeyError):
      acc.Images('hst1')

  def testNonValueEvents(self):
    """Tests that non-value events in the generator don't cause early exits."""
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    gen.AddScalar('sv1', wall_time=1, step=10, value=20)
    gen.AddEvent(tf.Event(
        wall_time=2, step=20, file_version='notsv2'))
    gen.AddScalar('sv3', wall_time=3, step=100, value=1)
    gen.AddHistogram('hst1')
    gen.AddImage('im1')

    acc.Reload()
    self.assertTagsEqual(acc.Tags(), {
        ea.IMAGES: ['im1'],
        ea.SCALARS: ['sv1', 'sv3'],
        ea.HISTOGRAMS: ['hst1'],
        ea.COMPRESSED_HISTOGRAMS: ['hst1'],
        ea.GRAPH: False})


class RealisticEventAccumulatorTest(EventAccumulatorTest):

  def setUp(self):
    super(RealisticEventAccumulatorTest, self).setUp()

  def testScalarsRealistically(self):
    """Test accumulator by writing values and then reading them."""
    def FakeScalarSummary(tag, value):
      value = tf.Summary.Value(tag=tag, simple_value=value)
      summary = tf.Summary(value=[value])
      return summary

    directory = os.path.join(self.get_temp_dir(), 'values_dir')
    if gfile.IsDirectory(directory):
      gfile.DeleteRecursively(directory)
    gfile.MkDir(directory)

    writer = tf.train.SummaryWriter(directory, max_queue=100)
    graph_def = tf.GraphDef(node=[tf.NodeDef(name='A', op='Mul')])
    # Add a graph to the summary writer.
    writer.add_graph(graph_def)

    # Write a bunch of events using the writer
    for i in xrange(30):
      summ_id = FakeScalarSummary('id', i)
      summ_sq = FakeScalarSummary('sq', i*i)
      writer.add_summary(summ_id, i*5)
      writer.add_summary(summ_sq, i*5)
    writer.flush()

    # Verify that we can load those events properly
    acc = ea.EventAccumulator(directory)
    acc.Reload()
    self.assertTagsEqual(acc.Tags(), {
        ea.IMAGES: [],
        ea.SCALARS: ['id', 'sq'],
        ea.HISTOGRAMS: [],
        ea.COMPRESSED_HISTOGRAMS: [],
        ea.GRAPH: True})
    id_events = acc.Scalars('id')
    sq_events = acc.Scalars('sq')
    self.assertEqual(30, len(id_events))
    self.assertEqual(30, len(sq_events))
    for i in xrange(30):
      self.assertEqual(i*5, id_events[i].step)
      self.assertEqual(i*5, sq_events[i].step)
      self.assertEqual(i, id_events[i].value)
      self.assertEqual(i*i, sq_events[i].value)

    # Write a few more events to test incremental reloading
    for i in xrange(30, 40):
      summ_id = FakeScalarSummary('id', i)
      summ_sq = FakeScalarSummary('sq', i*i)
      writer.add_summary(summ_id, i*5)
      writer.add_summary(summ_sq, i*5)
    writer.flush()

    # Verify we can now see all of the data
    acc.Reload()
    self.assertEqual(40, len(id_events))
    self.assertEqual(40, len(sq_events))
    for i in xrange(40):
      self.assertEqual(i*5, id_events[i].step)
      self.assertEqual(i*5, sq_events[i].step)
      self.assertEqual(i, id_events[i].value)
      self.assertEqual(i*i, sq_events[i].value)


if __name__ == '__main__':
  tf.test.main()
