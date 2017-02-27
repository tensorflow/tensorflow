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
"""Takes a generator of values, and accumulates them for a frontend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import threading

import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf.config_pb2 import RunMetadata
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary.impl import directory_watcher
from tensorflow.python.summary.impl import event_file_loader
from tensorflow.python.summary.impl import reservoir
from tensorflow.python.util import compat

namedtuple = collections.namedtuple
ScalarEvent = namedtuple('ScalarEvent', ['wall_time', 'step', 'value'])

HealthPillEvent = namedtuple(
    'HealthPillEvent',
    ['wall_time', 'step', 'node_name', 'output_slot', 'value'])

CompressedHistogramEvent = namedtuple('CompressedHistogramEvent',
                                      ['wall_time', 'step',
                                       'compressed_histogram_values'])

CompressedHistogramValue = namedtuple('CompressedHistogramValue',
                                      ['basis_point', 'value'])

HistogramEvent = namedtuple('HistogramEvent',
                            ['wall_time', 'step', 'histogram_value'])

HistogramValue = namedtuple('HistogramValue', ['min', 'max', 'num', 'sum',
                                               'sum_squares', 'bucket_limit',
                                               'bucket'])

ImageEvent = namedtuple('ImageEvent', ['wall_time', 'step',
                                       'encoded_image_string', 'width',
                                       'height'])

AudioEvent = namedtuple('AudioEvent', ['wall_time', 'step',
                                       'encoded_audio_string', 'content_type',
                                       'sample_rate', 'length_frames'])

TensorEvent = namedtuple('TensorEvent', ['wall_time', 'step', 'tensor_proto'])

## Different types of summary events handled by the event_accumulator
SUMMARY_TYPES = {
    'simple_value': '_ProcessScalar',
    'histo': '_ProcessHistogram',
    'image': '_ProcessImage',
    'audio': '_ProcessAudio',
    'tensor': '_ProcessTensor',
}

## The tagTypes below are just arbitrary strings chosen to pass the type
## information of the tag from the backend to the frontend
COMPRESSED_HISTOGRAMS = 'compressedHistograms'
HISTOGRAMS = 'histograms'
IMAGES = 'images'
AUDIO = 'audio'
SCALARS = 'scalars'
TENSORS = 'tensors'
HEALTH_PILLS = 'health_pills'
GRAPH = 'graph'
META_GRAPH = 'meta_graph'
RUN_METADATA = 'run_metadata'

## Normal CDF for std_devs: (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf)
## naturally gives bands around median of width 1 std dev, 2 std dev, 3 std dev,
## and then the long tail.
NORMAL_HISTOGRAM_BPS = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)

DEFAULT_SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 500,
    IMAGES: 4,
    AUDIO: 4,
    SCALARS: 10000,
    # We store this many health pills per op.
    HEALTH_PILLS: 100,
    HISTOGRAMS: 1,
    TENSORS: 10,
}

STORE_EVERYTHING_SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 0,
    IMAGES: 0,
    AUDIO: 0,
    SCALARS: 0,
    HEALTH_PILLS: 0,
    HISTOGRAMS: 0,
    TENSORS: 0,
}

# The tag that values containing health pills have. Health pill data is stored
# in tensors. In order to distinguish health pill values from scalar values, we
# rely on how health pill values have this special tag value.
_HEALTH_PILL_EVENT_TAG = '__health_pill__'


def IsTensorFlowEventsFile(path):
  """Check the path name to see if it is probably a TF Events file.

  Args:
    path: A file path to check if it is an event file.

  Raises:
    ValueError: If the path is an empty string.

  Returns:
    If path is formatted like a TensorFlowEventsFile.
  """
  if not path:
    raise ValueError('Path must be a nonempty string')
  return 'tfevents' in compat.as_str_any(os.path.basename(path))


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

  The `Reload()` method synchronously loads all of the data written so far.

  Histograms, audio, and images are very large, so storing all of them is not
  recommended.
  @@Tensors
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

    self._first_event_timestamp = None
    self._scalars = reservoir.Reservoir(size=sizes[SCALARS])

    # Unlike the other reservoir, the reservoir for health pills is keyed by the
    # name of the op instead of the tag. This lets us efficiently obtain the
    # health pills per node.
    self._health_pills = reservoir.Reservoir(size=sizes[HEALTH_PILLS])

    self._graph = None
    self._graph_from_metagraph = False
    self._meta_graph = None
    self._tagged_metadata = {}
    self._histograms = reservoir.Reservoir(size=sizes[HISTOGRAMS])
    self._compressed_histograms = reservoir.Reservoir(
        size=sizes[COMPRESSED_HISTOGRAMS], always_keep_last=False)
    self._images = reservoir.Reservoir(size=sizes[IMAGES])
    self._audio = reservoir.Reservoir(size=sizes[AUDIO])
    self._tensors = reservoir.Reservoir(size=sizes[TENSORS])

    self._generator_mutex = threading.Lock()
    self._generator = _GeneratorFromPath(path)

    self._compression_bps = compression_bps
    self.purge_orphaned_data = purge_orphaned_data

    self.most_recent_step = -1
    self.most_recent_wall_time = -1
    self.file_version = None

    # The attributes that get built up by the accumulator
    self.accumulated_attrs = ('_scalars', '_histograms',
                              '_compressed_histograms', '_images', '_audio')
    self._tensor_summaries = {}

  def Reload(self):
    """Loads all events added since the last call to `Reload`.

    If `Reload` was never called, loads all events in the file.

    Returns:
      The `EventAccumulator`.
    """
    with self._generator_mutex:
      for event in self._generator.Load():
        self._ProcessEvent(event)
    return self

  def FirstEventTimestamp(self):
    """Returns the timestamp in seconds of the first event.

    If the first event has been loaded (either by this method or by `Reload`,
    this returns immediately. Otherwise, it will load in the first event. Note
    that this means that calling `Reload` will cause this to block until
    `Reload` has finished.

    Returns:
      The timestamp in seconds of the first event that was loaded.

    Raises:
      ValueError: If no events have been loaded and there were no events found
      on disk.
    """
    if self._first_event_timestamp is not None:
      return self._first_event_timestamp
    with self._generator_mutex:
      try:
        event = next(self._generator.Load())
        self._ProcessEvent(event)
        return self._first_event_timestamp

      except StopIteration:
        raise ValueError('No event timestamp could be found')

  def _ProcessEvent(self, event):
    """Called whenever an event is loaded."""
    if self._first_event_timestamp is None:
      self._first_event_timestamp = event.wall_time

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

    ## Process the event.
    # GraphDef and MetaGraphDef are handled in a special way:
    # If no graph_def Event is available, but a meta_graph_def is, and it
    # contains a graph_def, then use the meta_graph_def.graph_def as our graph.
    # If a graph_def Event is available, always prefer it to the graph_def
    # inside the meta_graph_def.
    if event.HasField('graph_def'):
      if self._graph is not None:
        logging.warn(('Found more than one graph event per run, or there was '
                      'a metagraph containing a graph_def, as well as one or '
                      'more graph events.  Overwriting the graph with the '
                      'newest event.'))
      self._graph = event.graph_def
      self._graph_from_metagraph = False
    elif event.HasField('meta_graph_def'):
      if self._meta_graph is not None:
        logging.warn(('Found more than one metagraph event per run. '
                      'Overwriting the metagraph with the newest event.'))
      self._meta_graph = event.meta_graph_def
      if self._graph is None or self._graph_from_metagraph:
        # We may have a graph_def in the metagraph.  If so, and no
        # graph_def is directly available, use this one instead.
        meta_graph = meta_graph_pb2.MetaGraphDef()
        meta_graph.ParseFromString(self._meta_graph)
        if meta_graph.graph_def:
          if self._graph is not None:
            logging.warn(('Found multiple metagraphs containing graph_defs,'
                          'but did not find any graph events.  Overwriting the '
                          'graph with the newest metagraph version.'))
          self._graph_from_metagraph = True
          self._graph = meta_graph.graph_def.SerializeToString()
    elif event.HasField('tagged_run_metadata'):
      tag = event.tagged_run_metadata.tag
      if tag in self._tagged_metadata:
        logging.warn('Found more than one "run metadata" event with tag ' +
                     tag + '. Overwriting it with the newest event.')
      self._tagged_metadata[tag] = event.tagged_run_metadata.run_metadata
    elif event.HasField('summary'):
      for value in event.summary.value:
        if value.HasField('tensor') and value.tag == _HEALTH_PILL_EVENT_TAG:
          self._ProcessHealthPillSummary(value, event)
        else:
          for summary_type, summary_func in SUMMARY_TYPES.items():
            if value.HasField(summary_type):
              datum = getattr(value, summary_type)
              tag = value.node_name if summary_type == 'tensor' else value.tag
              getattr(self, summary_func)(tag, event.wall_time, event.step,
                                          datum)

  def _ProcessHealthPillSummary(self, value, event):
    """Process summaries containing health pills.

    These summaries are distinguished by the fact that they have a Tensor field
    and have a special tag value.

    This method emits ERROR-level messages to the logs if it encounters Tensor
    summaries that it cannot process.

    Args:
      value: A summary_pb2.Summary.Value with a Tensor field.
      event: The event_pb2.Event containing that value.
    """
    elements = np.fromstring(value.tensor.tensor_content, dtype=np.float64)

    # The node_name property of the value object is actually a watch key: a
    # combination of node name, output slot, and a suffix. We capture the
    # actual node name and the output slot with a regular expression.
    match = re.match(r'^(.*):(\d+):DebugNumericSummary$', value.node_name)
    if not match:
      logging.log_first_n(
          logging.ERROR,
          'Unsupported watch key %s for health pills; skipping this sequence.',
          1,
          value.node_name)
      return

    node_name = match.group(1)
    output_slot = int(match.group(2))
    self._ProcessHealthPill(
        event.wall_time, event.step, node_name, output_slot, elements)

  def Tags(self):
    """Return all tags found in the value stream.

    Returns:
      A `{tagType: ['list', 'of', 'tags']}` dictionary.
    """
    return {
        IMAGES: self._images.Keys(),
        AUDIO: self._audio.Keys(),
        HISTOGRAMS: self._histograms.Keys(),
        SCALARS: self._scalars.Keys(),
        COMPRESSED_HISTOGRAMS: self._compressed_histograms.Keys(),
        TENSORS: self._tensors.Keys(),
        # Use a heuristic: if the metagraph is available, but
        # graph is not, then we assume the metagraph contains the graph.
        GRAPH: self._graph is not None,
        META_GRAPH: self._meta_graph is not None,
        RUN_METADATA: list(self._tagged_metadata.keys())
    }

  def Scalars(self, tag):
    """Given a summary tag, return all associated `ScalarEvent`s.

    Args:
      tag: A string tag associated with the events.

    Raises:
      KeyError: If the tag is not found.

    Returns:
      An array of `ScalarEvent`s.
    """
    return self._scalars.Items(tag)

  def HealthPills(self, node_name):
    """Returns all health pill values for a certain node.

    Args:
      node_name: The name of the node to obtain health pills for.

    Raises:
      KeyError: If the node name is not found.

    Returns:
      An array of `HealthPillEvent`s.
    """
    return self._health_pills.Items(node_name)

  def Graph(self):
    """Return the graph definition, if there is one.

    If the graph is stored directly, return that.  If no graph is stored
    directly but a metagraph is stored containing a graph, return that.

    Raises:
      ValueError: If there is no graph for this run.

    Returns:
      The `graph_def` proto.
    """
    graph = graph_pb2.GraphDef()
    if self._graph is not None:
      graph.ParseFromString(self._graph)
      return graph
    raise ValueError('There is no graph in this EventAccumulator')

  def MetaGraph(self):
    """Return the metagraph definition, if there is one.

    Raises:
      ValueError: If there is no metagraph for this run.

    Returns:
      The `meta_graph_def` proto.
    """
    if self._meta_graph is None:
      raise ValueError('There is no metagraph in this EventAccumulator')
    meta_graph = meta_graph_pb2.MetaGraphDef()
    meta_graph.ParseFromString(self._meta_graph)
    return meta_graph

  def RunMetadata(self, tag):
    """Given a tag, return the associated session.run() metadata.

    Args:
      tag: A string tag associated with the event.

    Raises:
      ValueError: If the tag is not found.

    Returns:
      The metadata in form of `RunMetadata` proto.
    """
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

    Returns:
      An array of `HistogramEvent`s.
    """
    return self._histograms.Items(tag)

  def CompressedHistograms(self, tag):
    """Given a summary tag, return all associated compressed histograms.

    Args:
      tag: A string tag associated with the events.

    Raises:
      KeyError: If the tag is not found.

    Returns:
      An array of `CompressedHistogramEvent`s.
    """
    return self._compressed_histograms.Items(tag)

  def Images(self, tag):
    """Given a summary tag, return all associated images.

    Args:
      tag: A string tag associated with the events.

    Raises:
      KeyError: If the tag is not found.

    Returns:
      An array of `ImageEvent`s.
    """
    return self._images.Items(tag)

  def Audio(self, tag):
    """Given a summary tag, return all associated audio.

    Args:
      tag: A string tag associated with the events.

    Raises:
      KeyError: If the tag is not found.

    Returns:
      An array of `AudioEvent`s.
    """
    return self._audio.Items(tag)

  def Tensors(self, tag):
    """Given a summary tag, return all associated tensors.

    Args:
      tag: A string tag associated with the events.

    Raises:
      KeyError: If the tag is not found.

    Returns:
      An array of `TensorEvent`s.
    """
    return self._tensors.Items(tag)

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

  def _ConvertHistogramProtoToTuple(self, histo):
    return HistogramValue(min=histo.min,
                          max=histo.max,
                          num=histo.num,
                          sum=histo.sum,
                          sum_squares=histo.sum_squares,
                          bucket_limit=list(histo.bucket_limit),
                          bucket=list(histo.bucket))

  def _ProcessHistogram(self, tag, wall_time, step, histo):
    """Processes a proto histogram by adding it to accumulated state."""
    histo = self._ConvertHistogramProtoToTuple(histo)
    histo_ev = HistogramEvent(wall_time, step, histo)
    self._histograms.AddItem(tag, histo_ev)
    self._compressed_histograms.AddItem(
        tag, histo_ev, lambda x: _CompressHistogram(x, self._compression_bps))

  def _ProcessImage(self, tag, wall_time, step, image):
    """Processes an image by adding it to accumulated state."""
    event = ImageEvent(wall_time=wall_time,
                       step=step,
                       encoded_image_string=image.encoded_image_string,
                       width=image.width,
                       height=image.height)
    self._images.AddItem(tag, event)

  def _ProcessAudio(self, tag, wall_time, step, audio):
    """Processes a audio by adding it to accumulated state."""
    event = AudioEvent(wall_time=wall_time,
                       step=step,
                       encoded_audio_string=audio.encoded_audio_string,
                       content_type=audio.content_type,
                       sample_rate=audio.sample_rate,
                       length_frames=audio.length_frames)
    self._audio.AddItem(tag, event)

  def _ProcessScalar(self, tag, wall_time, step, scalar):
    """Processes a simple value by adding it to accumulated state."""
    sv = ScalarEvent(wall_time=wall_time, step=step, value=scalar)
    self._scalars.AddItem(tag, sv)

  def _ProcessTensor(self, tag, wall_time, step, tensor):
    tv = TensorEvent(wall_time=wall_time, step=step, tensor_proto=tensor)
    self._tensors.AddItem(tag, tv)

  def _ProcessHealthPill(self, wall_time, step, node_name, output_slot,
                         elements):
    """Processes a health pill value by adding it to accumulated state.

    Args:
      wall_time: The time at which the health pill was created. Provided by the
        debugger.
      step: The step at which the health pill was created. Provided by the
        debugger.
      node_name: The name of the node for this health pill.
      output_slot: The output slot for this health pill.
      elements: An ND array of 12 floats. The elements of the health pill.
    """
    # Key by the node name for fast retrieval of health pills by node name. The
    # array is cast to a list so that it is JSON-able. The debugger data plugin
    # serves a JSON response.
    self._health_pills.AddItem(
        node_name,
        HealthPillEvent(
            wall_time=wall_time,
            step=step,
            node_name=node_name,
            output_slot=output_slot,
            value=list(elements)))

  def _Purge(self, event, by_tags):
    """Purge all events that have occurred after the given event.step.

    If by_tags is True, purge all events that occurred after the given
    event.step, but only for the tags that the event has. Non-sequential
    event.steps suggest that a TensorFlow restart occurred, and we discard
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
                for x in self.accumulated_attrs]

      expired_per_tags = [_ExpiredPerTag(value)
                          for value in event.summary.value]
      expired_per_type = [sum(x) for x in zip(*expired_per_tags)]
    else:
      expired_per_type = [getattr(self, x).FilterItems(_NotExpired)
                          for x in self.accumulated_attrs]

    if sum(expired_per_type) > 0:
      purge_msg = _GetPurgeMessage(self.most_recent_step,
                                   self.most_recent_wall_time, event.step,
                                   event.wall_time, *expired_per_type)
      logging.warn(purge_msg)


def _GetPurgeMessage(most_recent_step, most_recent_wall_time, event_step,
                     event_wall_time, num_expired_scalars, num_expired_histos,
                     num_expired_comp_histos, num_expired_images,
                     num_expired_audio):
  """Return the string message associated with TensorBoard purges."""
  return ('Detected out of order event.step likely caused by '
          'a TensorFlow restart. Purging expired events from Tensorboard'
          ' display between the previous step: {} (timestamp: {}) and '
          'current step: {} (timestamp: {}). Removing {} scalars, {} '
          'histograms, {} compressed histograms, {} images, '
          'and {} audio.').format(most_recent_step, most_recent_wall_time,
                                  event_step, event_wall_time,
                                  num_expired_scalars, num_expired_histos,
                                  num_expired_comp_histos, num_expired_images,
                                  num_expired_audio)


def _GeneratorFromPath(path):
  """Create an event generator for file or directory at given path string."""
  if not path:
    raise ValueError('path must be a valid string')
  if IsTensorFlowEventsFile(path):
    return event_file_loader.EventFileLoader(path)
  else:
    return directory_watcher.DirectoryWatcher(
        path, event_file_loader.EventFileLoader, IsTensorFlowEventsFile)


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


def _CompressHistogram(histo_ev, bps):
  """Creates fixed size histogram by adding compression to accumulated state.

  This routine transforms a histogram at a particular step by linearly
  interpolating its variable number of buckets to represent their cumulative
  weight at a constant number of compression points. This significantly reduces
  the size of the histogram and makes it suitable for a two-dimensional area
  plot where the output of this routine constitutes the ranges for a single x
  coordinate.

  Args:
    histo_ev: A HistogramEvent namedtuple.
    bps: Compression points represented in basis points, 1/100ths of a percent.

  Returns:
    CompressedHistogramEvent namedtuple.
  """
  # See also: Histogram::Percentile() in core/lib/histogram/histogram.cc
  histo = histo_ev.histogram_value
  if not histo.num:
    return CompressedHistogramEvent(
        histo_ev.wall_time,
        histo_ev.step,
        [CompressedHistogramValue(b, 0.0) for b in bps])
  bucket = np.array(histo.bucket)
  weights = (bucket * bps[-1] / (bucket.sum() or 1.0)).cumsum()
  values = []
  j = 0
  while j < len(bps):
    i = np.searchsorted(weights, bps[j], side='right')
    while i < len(weights):
      cumsum = weights[i]
      cumsum_prev = weights[i - 1] if i > 0 else 0.0
      if cumsum == cumsum_prev:  # prevent remap divide by zero
        i += 1
        continue
      if not i or not cumsum_prev:
        lhs = histo.min
      else:
        lhs = max(histo.bucket_limit[i - 1], histo.min)
      rhs = min(histo.bucket_limit[i], histo.max)
      weight = _Remap(bps[j], cumsum_prev, cumsum, lhs, rhs)
      values.append(CompressedHistogramValue(bps[j], weight))
      j += 1
      break
    else:
      break
  while j < len(bps):
    values.append(CompressedHistogramValue(bps[j], histo.max))
    j += 1
  return CompressedHistogramEvent(histo_ev.wall_time, histo_ev.step, values)


def _Remap(x, x0, x1, y0, y1):
  """Linearly map from [x0, x1] unto [y0, y1]."""
  return y0 + (x - x0) * float(y1 - y0) / (x1 - x0)
