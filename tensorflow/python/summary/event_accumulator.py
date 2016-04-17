# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Takes a generator of values, and accumulates them for a frontend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import threading

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf.config_pb2 import RunMetadata
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.platform import logging
from tensorflow.python.summary.impl import directory_watcher
from tensorflow.python.summary.impl import io_wrapper
from tensorflow.python.summary.impl import reservoir

namedtuple = collections.namedtuple
ScalarEvent = namedtuple('ScalarEvent', ['wall_time', 'step', 'value'])

CompressedHistogramEvent = namedtuple('CompressedHistogramEvent',
                                      ['wall_time', 'step',
                                       'compressed_histogram_values'])

CompressedHistogramValue = namedtuple('CompressedHistogramValue',
                                      ['basis_point', 'value'])

HistogramEvent = namedtuple('HistogramEvent',
                            ['wall_time', 'step', 'histogram_value'])

HistogramValue = namedtuple('HistogramValue',
                            ['min', 'max', 'num', 'sum', 'sum_squares',
                             'bucket_limit', 'bucket'])

ImageEvent = namedtuple('ImageEvent',
                        ['wall_time', 'step', 'encoded_image_string', 'width',
                         'height'])

## Different types of summary events handled by the event_accumulator
SUMMARY_TYPES = ('_scalars', '_histograms', '_compressed_histograms', '_images')

## The tagTypes below are just arbitrary strings chosen to pass the type
## information of the tag from the backend to the frontend
COMPRESSED_HISTOGRAMS = 'compressedHistograms'
HISTOGRAMS = 'histograms'
IMAGES = 'images'
SCALARS = 'scalars'
GRAPH = 'graph'
RUN_METADATA = 'run_metadata'

## Normal CDF for std_devs: (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf)
## naturally gives bands around median of width 1 std dev, 2 std dev, 3 std dev,
## and then the long tail.
NORMAL_HISTOGRAM_BPS = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)

DEFAULT_SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 500,
    IMAGES: 4,
    SCALARS: 10000,
    HISTOGRAMS: 1,
}

STORE_EVERYTHING_SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 0,
    IMAGES: 0,
    SCALARS: 0,
    HISTOGRAMS: 0,
}


def IsTensorFlowEventsFile(path):
  """Check the path name to see if it is probably a TF Events file."""
  return 'tfevents' in os.path.basename(path)


class EventAccumulator(object):
  """An `EventAccumulator` takes an event generator, and accumulates the values.

  The `EventAccumulator` is intended to provide a convenient Python interface
  for loading Event data written during a TensorFlow run. TensorFlow writes out
  `Event` protobuf objects, which have a timestamp and step number, and often
  contain a `Summary`. Summaries can have different kinds of data like an image,
  a scalar value, or a histogram. The Summaries also have a tag, which we use to
  organize logically related data. The `EventAccumulator` supports retrieving
  the `Event` and `Summary` data by its tag.

  Calling `Tags()` gets a map from `tagType` (e.g. `'images'`,
  `'compressedHistograms'`, `'scalars'`, etc) to the associated tags for those
  data types. Then, various functional endpoints (eg
  `Accumulator.Scalars(tag)`) allow for the retrieval of all data
  associated with that tag.

  Before usage, the `EventAccumulator` must be activated via `Reload()`. This
  method synchronosly loads all of the data written so far.

  Histograms and images are very large, so storing all of them is not
  recommended.

  @@Reload
  @@Tags
  @@Scalars
  @@Graph
  @@RunMetadata
  @@Histograms
  @@CompressedHistograms
  @@Images
  """

  def __init__(self,
               path,
               size_guidance=DEFAULT_SIZE_GUIDANCE,
               compression_bps=NORMAL_HISTOGRAM_BPS,
               purge_orphaned_data=True):
    """Construct the `EventAccumulator`.

    Args:
      path: A file path to a directory containing tf events files, or a single
        tf events file. The accumulator will load events from this path.
      size_guidance: Information on how much data the EventAccumulator should
        store in memory. The DEFAULT_SIZE_GUIDANCE tries not to store too much
        so as to avoid OOMing the client. The size_guidance should be a map
        from a `tagType` string to an integer representing the number of
        items to keep per tag for items of that `tagType`. If the size is 0,
        all events are stored.
      compression_bps: Information on how the `EventAccumulator` should compress
        histogram data for the `CompressedHistograms` tag (for details see
        `ProcessCompressedHistogram`).
      purge_orphaned_data: Whether to discard any events that were "orphaned" by
        a TensorFlow restart.
    """
    sizes = {}
    for key in DEFAULT_SIZE_GUIDANCE:
      if key in size_guidance:
        sizes[key] = size_guidance[key]
      else:
        sizes[key] = DEFAULT_SIZE_GUIDANCE[key]

    self._scalars = reservoir.Reservoir(size=sizes[SCALARS])
    self._graph = None
    self._tagged_metadata = {}
    self._histograms = reservoir.Reservoir(size=sizes[HISTOGRAMS])
    self._compressed_histograms = reservoir.Reservoir(
        size=sizes[COMPRESSED_HISTOGRAMS])
    self._images = reservoir.Reservoir(size=sizes[IMAGES])

    self._generator_mutex = threading.Lock()
    self._generator = _GeneratorFromPath(path)

    self._compression_bps = compression_bps
    self.purge_orphaned_data = purge_orphaned_data

    self._activated = False
    self.most_recent_step = -1
    self.most_recent_wall_time = -1
    self.file_version = None

  def Reload(self):
    """Loads all events added since the last call to `Reload`.

    If `Reload` was never called, loads all events in the file.
    Calling `Reload` activates the `EventAccumulator`.

    Returns:
      The `EventAccumulator`.
    """
    self._activated = True
    with self._generator_mutex:
      for event in self._generator.Load():
        if event.HasField('file_version'):
          new_file_version = _ParseFileVersion(event.file_version)
          if self.file_version and self.file_version != new_file_version:
            ## This should not happen.
            logging.warn(('Found new file_version for event.proto. This will '
                          'affect purging logic for TensorFlow restarts. '
                          'Old: {0} New: {1}').format(self.file_version,
                                                      new_file_version))
          self.file_version = new_file_version

        self._MaybePurgeOrphanedData(event)

        ## Process the event
        if event.HasField('graph_def'):
          if self._graph is not None:
            logging.warn(('Found more than one graph event per run. '
                          'Overwriting the graph with the newest event.'))
          self._graph = event.graph_def
        elif event.HasField('tagged_run_metadata'):
          tag = event.tagged_run_metadata.tag
          if tag in self._tagged_metadata:
            logging.warn('Found more than one "run metadata" event with tag ' +
                         tag + '. Overwriting it with the newest event.')
          self._tagged_metadata[tag] = event.tagged_run_metadata.run_metadata
        elif event.HasField('summary'):
          for value in event.summary.value:
            if value.HasField('simple_value'):
              self._ProcessScalar(value.tag, event.wall_time, event.step,
                                  value.simple_value)
            elif value.HasField('histo'):
              self._ProcessHistogram(value.tag, event.wall_time, event.step,
                                     value.histo)
              self._ProcessCompressedHistogram(value.tag, event.wall_time,
                                               event.step, value.histo)
            elif value.HasField('image'):
              self._ProcessImage(value.tag, event.wall_time, event.step,
                                 value.image)
    return self

  def Tags(self):
    """Return all tags found in the value stream.

    Raises:
      RuntimeError: If the `EventAccumulator` has not been activated.

    Returns:
      A `{tagType: ['list', 'of', 'tags']}` dictionary.
    """
    self._VerifyActivated()
    return {IMAGES: self._images.Keys(),
            HISTOGRAMS: self._histograms.Keys(),
            SCALARS: self._scalars.Keys(),
            COMPRESSED_HISTOGRAMS: self._compressed_histograms.Keys(),
            GRAPH: self._graph is not None,
            RUN_METADATA: list(self._tagged_metadata.keys())}

  def Scalars(self, tag):
    """Given a summary tag, return all associated `ScalarEvent`s.

    Args:
      tag: A string tag associated with the events.

    Raises:
      KeyError: If the tag is not found.
      RuntimeError: If the `EventAccumulator` has not been activated.

    Returns:
      An array of `ScalarEvent`s.
    """
    self._VerifyActivated()
    return self._scalars.Items(tag)

  def Graph(self):
    """Return the graph definition, if there is one.

    Raises:
      ValueError: If there is no graph for this run.
      RuntimeError: If the `EventAccumulator` has not been activated.

    Returns:
      The `graph_def` proto.
    """
    self._VerifyActivated()
    if self._graph is None:
      raise ValueError('There is no graph in this EventAccumulator')
    graph = graph_pb2.GraphDef()
    graph.ParseFromString(self._graph)
    return graph

  def RunMetadata(self, tag):
    """Given a tag, return the associated session.run() metadata.

    Args:
      tag: A string tag associated with the event.

    Raises:
      ValueError: If the tag is not found.
      RuntimeError: If the `EventAccumulator` has not been activated.

    Returns:
      The metadata in form of `RunMetadata` proto.
    """
    self._VerifyActivated()
    if tag not in self._tagged_metadata:
      raise ValueError('There is no run metadata with this tag name')

    run_metadata = RunMetadata()
    run_metadata.ParseFromString(self._tagged_metadata[tag])
    return run_metadata

  def Histograms(self, tag):
    """Given a summary tag, return all associated histograms.

    Args:
      tag: A string tag associated with the events.

    Raises:
      KeyError: If the tag is not found.
      RuntimeError: If the `EventAccumulator` has not been activated.

    Returns:
      An array of `HistogramEvent`s.
    """
    self._VerifyActivated()
    return self._histograms.Items(tag)

  def CompressedHistograms(self, tag):
    """Given a summary tag, return all associated compressed histograms.

    Args:
      tag: A string tag associated with the events.

    Raises:
      KeyError: If the tag is not found.
      RuntimeError: If the `EventAccumulator` has not been activated.

    Returns:
      An array of `CompressedHistogramEvent`s.
    """
    self._VerifyActivated()
    return self._compressed_histograms.Items(tag)

  def Images(self, tag):
    """Given a summary tag, return all associated images.

    Args:
      tag: A string tag associated with the events.

    Raises:
      KeyError: If the tag is not found.
      RuntimeError: If the `EventAccumulator` has not been activated.

    Returns:
      An array of `ImageEvent`s.
    """
    self._VerifyActivated()
    return self._images.Items(tag)

  def _MaybePurgeOrphanedData(self, event):
    """Maybe purge orphaned data due to a TensorFlow crash.

    When TensorFlow crashes at step T+O and restarts at step T, any events
    written after step T are now "orphaned" and will be at best misleading if
    they are included in TensorBoard.

    This logic attempts to determine if there is orphaned data, and purge it
    if it is found.

    Args:
      event: The event to use as a reference, to determine if a purge is needed.
    """
    if not self.purge_orphaned_data:
      return
    ## Check if the event happened after a crash, and purge expired tags.
    if self.file_version and self.file_version >= 2:
      ## If the file_version is recent enough, use the SessionLog enum
      ## to check for restarts.
      self._CheckForRestartAndMaybePurge(event)
    else:
      ## If there is no file version, default to old logic of checking for
      ## out of order steps.
      self._CheckForOutOfOrderStepAndMaybePurge(event)

  def _CheckForRestartAndMaybePurge(self, event):
    """Check and discard expired events using SessionLog.START.

    Check for a SessionLog.START event and purge all previously seen events
    with larger steps, because they are out of date. Because of supervisor
    threading, it is possible that this logic will cause the first few event
    messages to be discarded since supervisor threading does not guarantee
    that the START message is deterministically written first.

    This method is preferred over _CheckForOutOfOrderStepAndMaybePurge which
    can inadvertently discard events due to supervisor threading.

    Args:
      event: The event to use as reference. If the event is a START event, all
        previously seen events with a greater event.step will be purged.
    """
    if event.HasField(
        'session_log') and event.session_log.status == SessionLog.START:
      self._Purge(event, by_tags=False)

  def _CheckForOutOfOrderStepAndMaybePurge(self, event):
    """Check for out-of-order event.step and discard expired events for tags.

    Check if the event is out of order relative to the global most recent step.
    If it is, purge outdated summaries for tags that the event contains.

    Args:
      event: The event to use as reference. If the event is out-of-order, all
        events with the same tags, but with a greater event.step will be purged.
    """
    if event.step < self.most_recent_step and event.HasField('summary'):
      self._Purge(event, by_tags=True)
    else:
      self.most_recent_step = event.step
      self.most_recent_wall_time = event.wall_time

  def _Percentile(self, compression_bps, bucket_limit, cumsum_weights,
                  histo_min, histo_max, histo_num):
    """Linearly interpolates a histogram weight for a particular basis point.

    Uses clamping methods on `histo_min` and `histo_max` to produce tight
    linear estimates of the histogram weight at a particular basis point.

    Args:
      compression_bps: The desired basis point at which to estimate the weight
      bucket_limit: An array of the RHS histogram bucket limits
      cumsum_weights: A cumulative sum of the fraction of weights in each
        histogram bucket, represented in basis points.
      histo_min: The minimum weight observed in the weight histogram
      histo_max: The maximum weight observed in the weight histogram
      histo_num: The number of items in the weight histogram

    Returns:
      A linearly interpolated value of the histogram weight estimate.
    """
    if histo_num == 0:
      return 0

    for i, cumsum in enumerate(cumsum_weights):
      if cumsum >= compression_bps:
        cumsum_prev = cumsum_weights[i - 1] if i > 0 else 0
        # Prevent cumsum = 0, cumsum_prev = 0, lerp divide by zero.
        if cumsum == cumsum_prev:
          continue

        # Calculate the lower bound of interpolation
        lhs = bucket_limit[i - 1] if (i > 0 and cumsum_prev > 0) else histo_min
        lhs = max(lhs, histo_min)

        # Calculate the upper bound of interpolation
        rhs = bucket_limit[i]
        rhs = min(rhs, histo_max)

        weight = _Remap(compression_bps, cumsum_prev, cumsum, lhs, rhs)
        return weight

    ## We have not exceeded cumsum, so return the max observed.
    return histo_max

  def _ProcessCompressedHistogram(self, tag, wall_time, step, histo):
    """Processes a histogram by adding a compression to accumulated state.

    Adds a compressed histogram by linearly interpolating histogram buckets to
    represent the histogram weight at multiple compression points. Uses
    self._compression_bps (passed to EventAccumulator constructor) as the
    compression points (represented in basis points, 1/100ths of a precent).

    Args:
      tag: A string name of the tag for which histograms are retrieved.
      wall_time: Time in seconds since epoch
      step: Number of steps that have passed
      histo: proto2 histogram Object
    """

    def _CumulativeSum(arr):
      return [sum(arr[:i + 1]) for i in range(len(arr))]

    # Convert from proto repeated field into a Python list.
    bucket = list(histo.bucket)
    bucket_limit = list(histo.bucket_limit)

    bucket_total = sum(bucket)
    if bucket_total == 0:
      bucket_total = 1
    fraction_weights = [10000 * x / bucket_total for x in bucket]
    cumsum_weights = _CumulativeSum(fraction_weights)

    percentiles = [
        self._Percentile(bps, bucket_limit, cumsum_weights, histo.min,
                         histo.max, histo.num) for bps in self._compression_bps
    ]

    compressed_histogram_values = [CompressedHistogramValue(
        basis_point=bps,
        value=value) for bps, value in zip(self._compression_bps, percentiles)]
    histogram_event = CompressedHistogramEvent(
        wall_time=wall_time,
        step=step,
        compressed_histogram_values=compressed_histogram_values)

    self._compressed_histograms.AddItem(tag, histogram_event)

  def _ProcessHistogram(self, tag, wall_time, step, histo):
    """Processes a histogram by adding it to accumulated state."""
    histogram_value = HistogramValue(min=histo.min,
                                     max=histo.max,
                                     num=histo.num,
                                     sum=histo.sum,
                                     sum_squares=histo.sum_squares,
                                     # Convert from proto repeated to list.
                                     bucket_limit=list(histo.bucket_limit),
                                     bucket=list(histo.bucket),)
    histogram_event = HistogramEvent(wall_time=wall_time,
                                     step=step,
                                     histogram_value=histogram_value,)
    self._histograms.AddItem(tag, histogram_event)

  def _ProcessImage(self, tag, wall_time, step, image):
    """Processes an image by adding it to accumulated state."""
    event = ImageEvent(wall_time=wall_time,
                       step=step,
                       encoded_image_string=image.encoded_image_string,
                       width=image.width,
                       height=image.height)
    self._images.AddItem(tag, event)

  def _ProcessScalar(self, tag, wall_time, step, scalar):
    """Processes a simple value by adding it to accumulated state."""
    sv = ScalarEvent(wall_time=wall_time, step=step, value=scalar)
    self._scalars.AddItem(tag, sv)

  def _Purge(self, event, by_tags):
    """Purge all events that have occurred after the given event.step.

    If by_tags is True, purge all events that occurred after the given
    event.step, but only for the tags that the event has. Non-sequential
    event.steps suggest that a Tensorflow restart occurred, and we discard
    the out-of-order events to display a consistent view in TensorBoard.

    Discarding by tags is the safer method, when we are unsure whether a restart
    has occurred, given that threading in supervisor can cause events of
    different tags to arrive with unsynchronized step values.

    If by_tags is False, then purge all events with event.step greater than the
    given event.step. This can be used when we are certain that a TensorFlow
    restart has occurred and these events can be discarded.

    Args:
      event: The event to use as reference for the purge. All events with
        the same tags, but with a greater event.step will be purged.
      by_tags: Bool to dictate whether to discard all out-of-order events or
        only those that are associated with the given reference event.
    """
    ## Keep data in reservoirs that has a step less than event.step
    _NotExpired = lambda x: x.step < event.step

    if by_tags:

      def _ExpiredPerTag(value):
        return [getattr(self, x).FilterItems(_NotExpired, value.tag)
                for x in SUMMARY_TYPES]

      expired_per_tags = [_ExpiredPerTag(value)
                          for value in event.summary.value]
      expired_per_type = [sum(x) for x in zip(*expired_per_tags)]
    else:
      expired_per_type = [getattr(self, x).FilterItems(_NotExpired)
                          for x in SUMMARY_TYPES]

    if sum(expired_per_type) > 0:
      purge_msg = _GetPurgeMessage(self.most_recent_step,
                                   self.most_recent_wall_time, event.step,
                                   event.wall_time, *expired_per_type)
      logging.warn(purge_msg)

  def _VerifyActivated(self):
    if not self._activated:
      raise RuntimeError('Accumulator must be activated before it may be used.')


def _GetPurgeMessage(most_recent_step, most_recent_wall_time, event_step,
                     event_wall_time, num_expired_scalars, num_expired_histos,
                     num_expired_comp_histos, num_expired_images):
  """Return the string message associated with TensorBoard purges."""
  return ('Detected out of order event.step likely caused by '
          'a TensorFlow restart. Purging expired events from Tensorboard'
          ' display between the previous step: {} (timestamp: {}) and '
          'current step: {} (timestamp: {}). Removing {} scalars, {} '
          'histograms, {} compressed histograms, and {} images.').format(
              most_recent_step, most_recent_wall_time, event_step,
              event_wall_time, num_expired_scalars, num_expired_histos,
              num_expired_comp_histos, num_expired_images)


def _GeneratorFromPath(path):
  """Create an event generator for file or directory at given path string."""
  if IsTensorFlowEventsFile(path):
    return io_wrapper.CreateFileLoader(path)
  else:
    provider = directory_watcher.SequentialFileProvider(
        path,
        path_filter=IsTensorFlowEventsFile)
    return directory_watcher.DirectoryWatcher(provider,
                                              io_wrapper.CreateFileLoader)


def _ParseFileVersion(file_version):
  """Convert the string file_version in event.proto into a float.

  Args:
    file_version: String file_version from event.proto

  Returns:
    Version number as a float.
  """
  tokens = file_version.split('brain.Event:')
  try:
    return float(tokens[-1])
  except ValueError:
    ## This should never happen according to the definition of file_version
    ## specified in event.proto.
    logging.warn(('Invalid event.proto file_version. Defaulting to use of '
                  'out-of-order event.step logic for purging expired events.'))
    return -1


def _Remap(x, x0, x1, y0, y1):
  """Linearly map from [x0, x1] unto [y0, y1]."""
  return y0 + (x - x0) * float(y1 - y0) / (x1 - x0)
