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

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_accumulator as ea


class _EventGenerator(object):

  def __init__(self):
    self.items = []

  def Load(self):
    while self.items:
      yield self.items.pop(0)

  def AddScalar(self, tag, wall_time=0, step=0, value=0):
    event = tf.Event(
        wall_time=wall_time,
        step=step,
        summary=tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                   simple_value=value)]))
    self.AddEvent(event)

  def AddHistogram(self,
                   tag,
                   wall_time=0,
                   step=0,
                   hmin=1,
                   hmax=2,
                   hnum=3,
                   hsum=4,
                   hsum_squares=5,
                   hbucket_limit=None,
                   hbucket=None):
    histo = tf.HistogramProto(min=hmin,
                              max=hmax,
                              num=hnum,
                              sum=hsum,
                              sum_squares=hsum_squares,
                              bucket_limit=hbucket_limit,
                              bucket=hbucket)
    event = tf.Event(wall_time=wall_time,
                     step=step,
                     summary=tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                                histo=histo)]))
    self.AddEvent(event)

  def AddImage(self,
               tag,
               wall_time=0,
               step=0,
               encoded_image_string=b'imgstr',
               width=150,
               height=100):
    image = tf.Summary.Image(encoded_image_string=encoded_image_string,
                             width=width,
                             height=height)
    event = tf.Event(wall_time=wall_time,
                     step=step,
                     summary=tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                                image=image)]))
    self.AddEvent(event)

  def AddAudio(self,
               tag,
               wall_time=0,
               step=0,
               encoded_audio_string=b'sndstr',
               content_type='audio/wav',
               sample_rate=44100,
               length_frames=22050):
    audio = tf.Summary.Audio(encoded_audio_string=encoded_audio_string,
                             content_type=content_type,
                             sample_rate=sample_rate,
                             length_frames=length_frames)
    event = tf.Event(wall_time=wall_time,
                     step=step,
                     summary=tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                                audio=audio)]))
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
    self.stubs = googletest.StubOutForTesting()
    self.empty = {ea.IMAGES: [],
                  ea.AUDIO: [],
                  ea.SCALARS: [],
                  ea.HISTOGRAMS: [],
                  ea.COMPRESSED_HISTOGRAMS: [],
                  ea.GRAPH: False,
                  ea.RUN_METADATA: []}
    self._real_constructor = ea.EventAccumulator
    self._real_generator = ea._GeneratorFromPath

    def _FakeAccumulatorConstructor(generator, *args, **kwargs):
      ea._GeneratorFromPath = lambda x: generator
      return self._real_constructor(generator, *args, **kwargs)

    ea.EventAccumulator = _FakeAccumulatorConstructor

  def tearDown(self):
    self.stubs.CleanUp()
    ea.EventAccumulator = self._real_constructor
    ea._GeneratorFromPath = self._real_generator

  def testEmptyAccumulator(self):
    gen = _EventGenerator()
    x = ea.EventAccumulator(gen)
    x.Reload()
    self.assertEqual(x.Tags(), self.empty)

  def testTags(self):
    gen = _EventGenerator()
    gen.AddScalar('s1')
    gen.AddScalar('s2')
    gen.AddHistogram('hst1')
    gen.AddHistogram('hst2')
    gen.AddImage('im1')
    gen.AddImage('im2')
    gen.AddAudio('snd1')
    gen.AddAudio('snd2')
    acc = ea.EventAccumulator(gen)
    acc.Reload()
    self.assertTagsEqual(acc.Tags(), {
        ea.IMAGES: ['im1', 'im2'],
        ea.AUDIO: ['snd1', 'snd2'],
        ea.SCALARS: ['s1', 's2'],
        ea.HISTOGRAMS: ['hst1', 'hst2'],
        ea.COMPRESSED_HISTOGRAMS: ['hst1', 'hst2'],
        ea.GRAPH: False,
        ea.RUN_METADATA: []
    })

  def testReload(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    acc.Reload()
    self.assertEqual(acc.Tags(), self.empty)
    gen.AddScalar('s1')
    gen.AddScalar('s2')
    gen.AddHistogram('hst1')
    gen.AddHistogram('hst2')
    gen.AddImage('im1')
    gen.AddImage('im2')
    gen.AddAudio('snd1')
    gen.AddAudio('snd2')
    acc.Reload()
    self.assertTagsEqual(acc.Tags(), {
        ea.IMAGES: ['im1', 'im2'],
        ea.AUDIO: ['snd1', 'snd2'],
        ea.SCALARS: ['s1', 's2'],
        ea.HISTOGRAMS: ['hst1', 'hst2'],
        ea.COMPRESSED_HISTOGRAMS: ['hst1', 'hst2'],
        ea.GRAPH: False,
        ea.RUN_METADATA: []
    })

  def testScalars(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    s1 = ea.ScalarEvent(wall_time=1, step=10, value=32)
    s2 = ea.ScalarEvent(wall_time=2, step=12, value=64)
    gen.AddScalar('s1', wall_time=1, step=10, value=32)
    gen.AddScalar('s2', wall_time=2, step=12, value=64)
    acc.Reload()
    self.assertEqual(acc.Scalars('s1'), [s1])
    self.assertEqual(acc.Scalars('s2'), [s2])

  def testHistograms(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)

    val1 = ea.HistogramValue(min=1,
                             max=2,
                             num=3,
                             sum=4,
                             sum_squares=5,
                             bucket_limit=[1, 2, 3],
                             bucket=[0, 3, 0])
    val2 = ea.HistogramValue(min=-2,
                             max=3,
                             num=4,
                             sum=5,
                             sum_squares=6,
                             bucket_limit=[2, 3, 4],
                             bucket=[1, 3, 0])

    hst1 = ea.HistogramEvent(wall_time=1, step=10, histogram_value=val1)
    hst2 = ea.HistogramEvent(wall_time=2, step=12, histogram_value=val2)
    gen.AddHistogram('hst1',
                     wall_time=1,
                     step=10,
                     hmin=1,
                     hmax=2,
                     hnum=3,
                     hsum=4,
                     hsum_squares=5,
                     hbucket_limit=[1, 2, 3],
                     hbucket=[0, 3, 0])
    gen.AddHistogram('hst2',
                     wall_time=2,
                     step=12,
                     hmin=-2,
                     hmax=3,
                     hnum=4,
                     hsum=5,
                     hsum_squares=6,
                     hbucket_limit=[2, 3, 4],
                     hbucket=[1, 3, 0])
    acc.Reload()
    self.assertEqual(acc.Histograms('hst1'), [hst1])
    self.assertEqual(acc.Histograms('hst2'), [hst2])

  def testCompressedHistograms(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen, compression_bps=(0, 2500, 5000, 7500, 10000))

    gen.AddHistogram('hst1',
                     wall_time=1,
                     step=10,
                     hmin=1,
                     hmax=2,
                     hnum=3,
                     hsum=4,
                     hsum_squares=5,
                     hbucket_limit=[1, 2, 3],
                     hbucket=[0, 3, 0])
    gen.AddHistogram('hst2',
                     wall_time=2,
                     step=12,
                     hmin=-2,
                     hmax=3,
                     hnum=4,
                     hsum=5,
                     hsum_squares=6,
                     hbucket_limit=[2, 3, 4],
                     hbucket=[1, 3, 0])
    acc.Reload()

    # Create the expected values after compressing hst1
    expected_vals1 = [ea.CompressedHistogramValue(bp, val)
                      for bp, val in [(0, 1.0), (2500, 1.25), (5000, 1.5), (
                          7500, 1.75), (10000, 2.0)]]
    expected_cmphst1 = ea.CompressedHistogramEvent(
        wall_time=1,
        step=10,
        compressed_histogram_values=expected_vals1)
    self.assertEqual(acc.CompressedHistograms('hst1'), [expected_cmphst1])

    # Create the expected values after compressing hst2
    expected_vals2 = [
        ea.CompressedHistogramValue(bp, val)
        for bp, val in [(0, -2), (2500, 2), (5000, 2 + 1 / 3), (7500, 2 + 2 / 3
                                                               ), (10000, 3)]
    ]
    expected_cmphst2 = ea.CompressedHistogramEvent(
        wall_time=2,
        step=12,
        compressed_histogram_values=expected_vals2)
    self.assertEqual(acc.CompressedHistograms('hst2'), [expected_cmphst2])

  def testPercentile(self):

    def AssertExpectedForBps(bps, expected):
      output = acc._Percentile(bps, bucket_limit, cumsum_weights, histo_min,
                               histo_max, histo_num)
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
    AssertExpectedForBps(2500, ea._Remap(2500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(5000, ea._Remap(5000, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(7500, ea._Remap(7500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## All weights in second bucket
    cumsum_weights = [0, 10000, 10000, 10000]
    histo_min = 1.1
    histo_max = 1.8
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, ea._Remap(2500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(5000, ea._Remap(5000, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(7500, ea._Remap(7500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## All weights in the last bucket
    cumsum_weights = [0, 0, 0, 10000]
    histo_min = 3.1
    histo_max = 3.6
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, ea._Remap(2500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(5000, ea._Remap(5000, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(7500, ea._Remap(7500, 0, 10000, histo_min, histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## Weights distributed between two buckets
    cumsum_weights = [0, 4000, 10000, 10000]
    histo_min = 1.1
    histo_max = 2.9
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, ea._Remap(2500, 0, 4000, histo_min,
                                         bucket_limit[1]))
    AssertExpectedForBps(5000, ea._Remap(5000, 4000, 10000, bucket_limit[1],
                                         histo_max))
    AssertExpectedForBps(7500, ea._Remap(7500, 4000, 10000, bucket_limit[1],
                                         histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## Weights distributed between all buckets
    cumsum_weights = [1000, 4000, 8000, 10000]
    histo_min = -1
    histo_max = 3.9
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, ea._Remap(2500, 1000, 4000, bucket_limit[0],
                                         bucket_limit[1]))
    AssertExpectedForBps(5000, ea._Remap(5000, 4000, 8000, bucket_limit[1],
                                         bucket_limit[2]))
    AssertExpectedForBps(7500, ea._Remap(7500, 4000, 8000, bucket_limit[1],
                                         bucket_limit[2]))
    AssertExpectedForBps(9000, ea._Remap(9000, 8000, 10000, bucket_limit[2],
                                         histo_max))
    AssertExpectedForBps(10000, histo_max)

    ## Most weight in first bucket
    cumsum_weights = [9000, 10000, 10000, 10000]
    histo_min = -1
    histo_max = 1.1
    AssertExpectedForBps(0, histo_min)
    AssertExpectedForBps(2500, ea._Remap(2500, 0, 9000, histo_min,
                                         bucket_limit[0]))
    AssertExpectedForBps(5000, ea._Remap(5000, 0, 9000, histo_min,
                                         bucket_limit[0]))
    AssertExpectedForBps(7500, ea._Remap(7500, 0, 9000, histo_min,
                                         bucket_limit[0]))
    AssertExpectedForBps(9500, ea._Remap(9500, 9000, 10000, bucket_limit[0],
                                         histo_max))
    AssertExpectedForBps(10000, histo_max)

  def testImages(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    im1 = ea.ImageEvent(wall_time=1,
                        step=10,
                        encoded_image_string=b'big',
                        width=400,
                        height=300)
    im2 = ea.ImageEvent(wall_time=2,
                        step=12,
                        encoded_image_string=b'small',
                        width=40,
                        height=30)
    gen.AddImage('im1',
                 wall_time=1,
                 step=10,
                 encoded_image_string=b'big',
                 width=400,
                 height=300)
    gen.AddImage('im2',
                 wall_time=2,
                 step=12,
                 encoded_image_string=b'small',
                 width=40,
                 height=30)
    acc.Reload()
    self.assertEqual(acc.Images('im1'), [im1])
    self.assertEqual(acc.Images('im2'), [im2])

  def testAudio(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    snd1 = ea.AudioEvent(wall_time=1,
                         step=10,
                         encoded_audio_string=b'big',
                         content_type='audio/wav',
                         sample_rate=44100,
                         length_frames=441000)
    snd2 = ea.AudioEvent(wall_time=2,
                         step=12,
                         encoded_audio_string=b'small',
                         content_type='audio/wav',
                         sample_rate=44100,
                         length_frames=44100)
    gen.AddAudio('snd1',
                 wall_time=1,
                 step=10,
                 encoded_audio_string=b'big',
                 content_type='audio/wav',
                 sample_rate=44100,
                 length_frames=441000)
    gen.AddAudio('snd2',
                 wall_time=2,
                 step=12,
                 encoded_audio_string=b'small',
                 content_type='audio/wav',
                 sample_rate=44100,
                 length_frames=44100)
    acc.Reload()
    self.assertEqual(acc.Audio('snd1'), [snd1])
    self.assertEqual(acc.Audio('snd2'), [snd2])

  def testKeyError(self):
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    acc.Reload()
    with self.assertRaises(KeyError):
      acc.Scalars('s1')
    with self.assertRaises(KeyError):
      acc.Scalars('hst1')
    with self.assertRaises(KeyError):
      acc.Scalars('im1')
    with self.assertRaises(KeyError):
      acc.Histograms('s1')
    with self.assertRaises(KeyError):
      acc.Histograms('im1')
    with self.assertRaises(KeyError):
      acc.Images('s1')
    with self.assertRaises(KeyError):
      acc.Images('hst1')
    with self.assertRaises(KeyError):
      acc.Audio('s1')
    with self.assertRaises(KeyError):
      acc.Audio('hst1')

  def testNonValueEvents(self):
    """Tests that non-value events in the generator don't cause early exits."""
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    gen.AddScalar('s1', wall_time=1, step=10, value=20)
    gen.AddEvent(tf.Event(wall_time=2, step=20, file_version='nots2'))
    gen.AddScalar('s3', wall_time=3, step=100, value=1)
    gen.AddHistogram('hst1')
    gen.AddImage('im1')
    gen.AddAudio('snd1')

    acc.Reload()
    self.assertTagsEqual(acc.Tags(), {
        ea.IMAGES: ['im1'],
        ea.AUDIO: ['snd1'],
        ea.SCALARS: ['s1', 's3'],
        ea.HISTOGRAMS: ['hst1'],
        ea.COMPRESSED_HISTOGRAMS: ['hst1'],
        ea.GRAPH: False,
        ea.RUN_METADATA: []
    })

  def testExpiredDataDiscardedAfterRestartForFileVersionLessThan2(self):
    """Tests that events are discarded after a restart is detected.

    If a step value is observed to be lower than what was previously seen,
    this should force a discard of all previous items with the same tag
    that are outdated.

    Only file versions < 2 use this out-of-order discard logic. Later versions
    discard events based on the step value of SessionLog.START.
    """
    warnings = []
    self.stubs.Set(logging, 'warn', warnings.append)

    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)

    gen.AddEvent(tf.Event(wall_time=0, step=0, file_version='brain.Event:1'))
    gen.AddScalar('s1', wall_time=1, step=100, value=20)
    gen.AddScalar('s1', wall_time=1, step=200, value=20)
    gen.AddScalar('s1', wall_time=1, step=300, value=20)
    acc.Reload()
    ## Check that number of items are what they should be
    self.assertEqual([x.step for x in acc.Scalars('s1')], [100, 200, 300])

    gen.AddScalar('s1', wall_time=1, step=101, value=20)
    gen.AddScalar('s1', wall_time=1, step=201, value=20)
    gen.AddScalar('s1', wall_time=1, step=301, value=20)
    acc.Reload()
    ## Check that we have discarded 200 and 300 from s1
    self.assertEqual([x.step for x in acc.Scalars('s1')], [100, 101, 201, 301])

  def testOrphanedDataNotDiscardedIfFlagUnset(self):
    """Tests that events are not discarded if purge_orphaned_data is false.
    """
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen, purge_orphaned_data=False)

    gen.AddEvent(tf.Event(wall_time=0, step=0, file_version='brain.Event:1'))
    gen.AddScalar('s1', wall_time=1, step=100, value=20)
    gen.AddScalar('s1', wall_time=1, step=200, value=20)
    gen.AddScalar('s1', wall_time=1, step=300, value=20)
    acc.Reload()
    ## Check that number of items are what they should be
    self.assertEqual([x.step for x in acc.Scalars('s1')], [100, 200, 300])

    gen.AddScalar('s1', wall_time=1, step=101, value=20)
    gen.AddScalar('s1', wall_time=1, step=201, value=20)
    gen.AddScalar('s1', wall_time=1, step=301, value=20)
    acc.Reload()
    ## Check that we have discarded 200 and 300 from s1
    self.assertEqual([x.step for x in acc.Scalars('s1')], [100, 200, 300, 101,
                                                           201, 301])

  def testEventsDiscardedPerTagAfterRestartForFileVersionLessThan2(self):
    """Tests that event discards after restart, only affect the misordered tag.

    If a step value is observed to be lower than what was previously seen,
    this should force a discard of all previous items that are outdated, but
    only for the out of order tag. Other tags should remain unaffected.

    Only file versions < 2 use this out-of-order discard logic. Later versions
    discard events based on the step value of SessionLog.START.
    """
    warnings = []
    self.stubs.Set(logging, 'warn', warnings.append)

    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)

    gen.AddEvent(tf.Event(wall_time=0, step=0, file_version='brain.Event:1'))
    gen.AddScalar('s1', wall_time=1, step=100, value=20)
    gen.AddScalar('s1', wall_time=1, step=200, value=20)
    gen.AddScalar('s1', wall_time=1, step=300, value=20)
    gen.AddScalar('s1', wall_time=1, step=101, value=20)
    gen.AddScalar('s1', wall_time=1, step=201, value=20)
    gen.AddScalar('s1', wall_time=1, step=301, value=20)

    gen.AddScalar('s2', wall_time=1, step=101, value=20)
    gen.AddScalar('s2', wall_time=1, step=201, value=20)
    gen.AddScalar('s2', wall_time=1, step=301, value=20)

    acc.Reload()
    ## Check that we have discarded 200 and 300
    self.assertEqual([x.step for x in acc.Scalars('s1')], [100, 101, 201, 301])

    ## Check that s1 discards do not affect s2
    ## i.e. check that only events from the out of order tag are discarded
    self.assertEqual([x.step for x in acc.Scalars('s2')], [101, 201, 301])

  def testOnlySummaryEventsTriggerDiscards(self):
    """Test that file version event does not trigger data purge."""
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    gen.AddScalar('s1', wall_time=1, step=100, value=20)
    ev1 = tf.Event(wall_time=2, step=0, file_version='brain.Event:1')
    graph_bytes = graph_pb2.GraphDef().SerializeToString()
    ev2 = tf.Event(wall_time=3, step=0, graph_def=graph_bytes)
    gen.AddEvent(ev1)
    gen.AddEvent(ev2)
    acc.Reload()
    self.assertEqual([x.step for x in acc.Scalars('s1')], [100])

  def testSessionLogStartMessageDiscardsExpiredEvents(self):
    """Test that SessionLog.START message discards expired events.

    This discard logic is preferred over the out-of-order step discard logic,
    but this logic can only be used for event protos which have the SessionLog
    enum, which was introduced to event.proto for file_version >= brain.Event:2.
    """
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    gen.AddEvent(tf.Event(wall_time=0, step=1, file_version='brain.Event:2'))

    gen.AddScalar('s1', wall_time=1, step=100, value=20)
    gen.AddScalar('s1', wall_time=1, step=200, value=20)
    gen.AddScalar('s1', wall_time=1, step=300, value=20)
    gen.AddScalar('s1', wall_time=1, step=400, value=20)

    gen.AddScalar('s2', wall_time=1, step=202, value=20)
    gen.AddScalar('s2', wall_time=1, step=203, value=20)

    slog = SessionLog(status=SessionLog.START)
    gen.AddEvent(tf.Event(wall_time=2, step=201, session_log=slog))
    acc.Reload()
    self.assertEqual([x.step for x in acc.Scalars('s1')], [100, 200])
    self.assertEqual([x.step for x in acc.Scalars('s2')], [])

  def testFirstEventTimestamp(self):
    """Test that FirstEventTimestamp() returns wall_time of the first event."""
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    gen.AddEvent(tf.Event(wall_time=10, step=20, file_version='brain.Event:2'))
    gen.AddScalar('s1', wall_time=30, step=40, value=20)
    self.assertEqual(acc.FirstEventTimestamp(), 10)

  def testReloadPopulatesFirstEventTimestamp(self):
    """Test that Reload() means FirstEventTimestamp() won't load events."""
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    gen.AddEvent(tf.Event(wall_time=1, step=2, file_version='brain.Event:2'))

    acc.Reload()

    def _Die(*args, **kwargs):  # pylint: disable=unused-argument
      raise RuntimeError('Load() should not be called')

    self.stubs.Set(gen, 'Load', _Die)
    self.assertEqual(acc.FirstEventTimestamp(), 1)

  def testFirstEventTimestampLoadsEvent(self):
    """Test that FirstEventTimestamp() doesn't discard the loaded event."""
    gen = _EventGenerator()
    acc = ea.EventAccumulator(gen)
    gen.AddEvent(tf.Event(wall_time=1, step=2, file_version='brain.Event:2'))

    self.assertEqual(acc.FirstEventTimestamp(), 1)
    acc.Reload()
    self.assertEqual(acc.file_version, 2.0)


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

    with tf.Graph().as_default() as graph:
      _ = tf.constant([2.0, 1.0])
    # Add a graph to the summary writer.
    writer.add_graph(graph)

    run_metadata = tf.RunMetadata()
    device_stats = run_metadata.step_stats.dev_stats.add()
    device_stats.device = 'test device'
    writer.add_run_metadata(run_metadata, 'test run')

    # Write a bunch of events using the writer.
    for i in xrange(30):
      summ_id = FakeScalarSummary('id', i)
      summ_sq = FakeScalarSummary('sq', i * i)
      writer.add_summary(summ_id, i * 5)
      writer.add_summary(summ_sq, i * 5)
    writer.flush()

    # Verify that we can load those events properly
    acc = ea.EventAccumulator(directory)
    acc.Reload()
    self.assertTagsEqual(
        acc.Tags(),
        {
            ea.IMAGES: [],
            ea.AUDIO: [],
            ea.SCALARS: ['id', 'sq'],
            ea.HISTOGRAMS: [],
            ea.COMPRESSED_HISTOGRAMS: [],
            ea.GRAPH: True,
            ea.RUN_METADATA: ['test run']
        })
    id_events = acc.Scalars('id')
    sq_events = acc.Scalars('sq')
    self.assertEqual(30, len(id_events))
    self.assertEqual(30, len(sq_events))
    for i in xrange(30):
      self.assertEqual(i * 5, id_events[i].step)
      self.assertEqual(i * 5, sq_events[i].step)
      self.assertEqual(i, id_events[i].value)
      self.assertEqual(i * i, sq_events[i].value)

    # Write a few more events to test incremental reloading
    for i in xrange(30, 40):
      summ_id = FakeScalarSummary('id', i)
      summ_sq = FakeScalarSummary('sq', i * i)
      writer.add_summary(summ_id, i * 5)
      writer.add_summary(summ_sq, i * 5)
    writer.flush()

    # Verify we can now see all of the data
    acc.Reload()
    self.assertEqual(40, len(id_events))
    self.assertEqual(40, len(sq_events))
    for i in xrange(40):
      self.assertEqual(i * 5, id_events[i].step)
      self.assertEqual(i * 5, sq_events[i].step)
      self.assertEqual(i, id_events[i].value)
      self.assertEqual(i * i, sq_events[i].value)
    self.assertProtoEquals(graph.as_graph_def(add_shapes=True), acc.Graph())


if __name__ == '__main__':
  tf.test.main()
