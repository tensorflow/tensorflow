"""Takes a generator of values, and accumulates them for a frontend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import threading

from tensorflow.python.platform import gfile
from tensorflow.python.platform import logging
from tensorflow.python.summary.impl import directory_watcher
from tensorflow.python.summary.impl import event_file_loader
from tensorflow.python.summary.impl import reservoir

namedtuple = collections.namedtuple
ScalarEvent = namedtuple('ScalarEvent',
                         ['wall_time', 'step', 'value'])

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
                        ['wall_time', 'step', 'encoded_image_string',
                         'width', 'height'])

## The tagTypes below are just arbitrary strings chosen to pass the type
## information of the tag from the backend to the frontend
COMPRESSED_HISTOGRAMS = 'compressedHistograms'
HISTOGRAMS = 'histograms'
IMAGES = 'images'
SCALARS = 'scalars'
GRAPH = 'graph'

## normal CDF for std_devs: (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf)
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
  return 'tfevents' in path


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

  Before usage, the `EventAccumulator` must be activated via `Reload()` or
  `AutoUpdate(interval)`.

  If activated via `Reload()`, it loads synchronously, so calls to `Values` or
  `Tags` will block until all outstanding events are processed. Afterwards,
  `Reload()` may be called again to load any new data.

  If activated via `AutoUpdate(interval)`, it loads asynchronously, so calls to
  `Values` or `Tags` will immediately return a valid subset of the outstanding
  event data. It reloads new data every `interval` seconds.

  Histograms and images are very large, so storing all of them is not
  recommended.

  @@Reload
  @@AutoUpdate
  @@Tags
  @@Scalars
  @@Graph
  @@Histograms
  @@CompressedHistograms
  @@Images
  """

  def __init__(self, path, size_guidance=DEFAULT_SIZE_GUIDANCE,
               compression_bps=NORMAL_HISTOGRAM_BPS):
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
    """
    sizes = {}
    for key in DEFAULT_SIZE_GUIDANCE:
      if key in size_guidance:
        sizes[key] = size_guidance[key]
      else:
        sizes[key] = DEFAULT_SIZE_GUIDANCE[key]

    self._scalars = reservoir.Reservoir(size=sizes[SCALARS])
    self._graph = None
    self._histograms = reservoir.Reservoir(size=sizes[HISTOGRAMS])
    self._compressed_histograms = reservoir.Reservoir(
        size=sizes[COMPRESSED_HISTOGRAMS])
    self._images = reservoir.Reservoir(size=sizes[IMAGES])
    self._generator_mutex = threading.Lock()
    self._generator = _GeneratorFromPath(path)
    self._is_autoupdating = False
    self._activated = False
    self._compression_bps = compression_bps
    self.most_recent_step = -1
    self.most_recent_wall_time = -1

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
        ## Check if the event happened after a crash
        if event.step < self.most_recent_step:

          ## Keep data in reservoirs that has a step less than event.step
          _NotExpired = lambda x: x.step < event.step
          num_expired_scalars = self._scalars.FilterItems(_NotExpired)
          num_expired_histograms = self._histograms.FilterItems(_NotExpired)
          num_expired_compressed_histograms = self._compressed_histograms.FilterItems(
              _NotExpired)
          num_expired_images = self._images.FilterItems(_NotExpired)

          purge_msg = (
              'Detected out of order event.step likely caused by a Tensorflow '
              'restart. Purging expired events from Tensorboard display '
              'between the previous step: {} (timestamp: {}) and current step:'
              ' {} (timestamp: {}). Removing {} scalars, {} histograms, {} '
              'compressed histograms, and {} images.').format(
                  self.most_recent_step, self.most_recent_wall_time, event.step,
                  event.wall_time, num_expired_scalars, num_expired_histograms,
                  num_expired_compressed_histograms, num_expired_images)
          logging.warn(purge_msg)
        else:
          self.most_recent_step = event.step
          self.most_recent_wall_time = event.wall_time
        ## Process the event
        if event.HasField('graph_def'):
          if self._graph is not None:
            logging.warn(('Found more than one graph event per run.'
                          'Overwritting the graph with the newest event'))
          self._graph = event.graph_def
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

  def AutoUpdate(self, interval=60):
    """Asynchronously load all events, and periodically reload.

    Calling this function is not thread safe.
    Calling this function activates the `EventAccumulator`.

    Args:
      interval: how many seconds after each successful reload to load new events
        (default 60)

    Returns:
      The `EventAccumulator`.
    """
    if self._is_autoupdating:
      return
    self._is_autoupdating = True
    self._activated = True
    def Update():
      self.Reload()
      logging.info('EventAccumulator update triggered')
      t = threading.Timer(interval, Update)
      t.daemon = True
      t.start()
    # Asynchronously start the update process, so that the accumulator can
    # immediately serve data, even if there is a very large event file to parse
    t = threading.Timer(0, Update)
    t.daemon = True
    t.start()
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
            GRAPH: self._graph is not None}

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
    return self._graph

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

  def _VerifyActivated(self):
    if not self._activated:
      raise RuntimeError('Accumulator must be activated before it may be used.')

  def _ProcessScalar(self, tag, wall_time, step, scalar):
    """Processes a simple value by adding it to accumulated state."""
    sv = ScalarEvent(wall_time=wall_time, step=step, value=scalar)
    self._scalars.AddItem(tag, sv)

  def _ProcessHistogram(self, tag, wall_time, step, histo):
    """Processes a histogram by adding it to accumulated state."""
    histogram_value = HistogramValue(
        min=histo.min,
        max=histo.max,
        num=histo.num,
        sum=histo.sum,
        sum_squares=histo.sum_squares,
        # convert from proto repeated to list
        bucket_limit=list(histo.bucket_limit),
        bucket=list(histo.bucket),
    )
    histogram_event = HistogramEvent(
        wall_time=wall_time,
        step=step,
        histogram_value=histogram_value,
    )
    self._histograms.AddItem(tag, histogram_event)

  def _Remap(self, x, x0, x1, y0, y1):
    """Linearly map from [x0, x1] unto [y0, y1]."""
    return y0 + (x - x0) * float(y1 - y0)/(x1 - x0)

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
    if histo_num == 0: return 0

    for i, cumsum in enumerate(cumsum_weights):
      if cumsum >= compression_bps:
        cumsum_prev = cumsum_weights[i-1] if i > 0 else 0
        # Prevent cumsum = 0, cumsum_prev = 0, lerp divide by zero.
        if cumsum == cumsum_prev: continue

        # Calculate the lower bound of interpolation
        lhs = bucket_limit[i-1] if (i > 0 and cumsum_prev > 0) else histo_min
        lhs = max(lhs, histo_min)

        # Calculate the upper bound of interpolation
        rhs = bucket_limit[i]
        rhs = min(rhs, histo_max)

        weight = self._Remap(compression_bps, cumsum_prev, cumsum, lhs, rhs)
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
      return [sum(arr[:i+1]) for i in range(len(arr))]

    # Convert from proto repeated field into a Python list.
    bucket = list(histo.bucket)
    bucket_limit = list(histo.bucket_limit)

    bucket_total = sum(bucket)
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

  def _ProcessImage(self, tag, wall_time, step, image):
    """Processes an image by adding it to accumulated state."""
    event = ImageEvent(
        wall_time=wall_time,
        step=step,
        encoded_image_string=image.encoded_image_string,
        width=image.width,
        height=image.height
    )
    self._images.AddItem(tag, event)


def _GeneratorFromPath(path):
  """Create an event generator for file or directory at given path string."""
  loader_factory = event_file_loader.EventFileLoader
  if gfile.IsDirectory(path):
    return directory_watcher.DirectoryWatcher(path, loader_factory,
                                              IsTensorFlowEventsFile)
  else:
    return loader_factory(path)
