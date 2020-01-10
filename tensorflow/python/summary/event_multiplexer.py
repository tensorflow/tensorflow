"""Provides an interface for working with multiple event files."""

import os
import threading

from tensorflow.python.platform import gfile
from tensorflow.python.platform import logging
from tensorflow.python.summary import event_accumulator


class EventMultiplexer(object):
  """An `EventMultiplexer` manages access to multiple `EventAccumulator`s.

  Each `EventAccumulator` is associated with a `run`, which is a self-contained
  TensorFlow execution. The `EventMultiplexer` provides methods for extracting
  information about events from multiple `run`s.

  Example usage for loading specific runs from files:

  ```python
  x = EventMultiplexer({'run1': 'path/to/run1', 'run2': 'path/to/run2'})
  x.Reload()
  ```

  Example usage for loading a directory where each subdirectory is a run

  ```python
  (eg:) /parent/directory/path/
        /parent/directory/path/run1/
        /parent/directory/path/run1/events.out.tfevents.1001
        /parent/directory/path/run1/events.out.tfevents.1002

        /parent/directory/path/run2/
        /parent/directory/path/run2/events.out.tfevents.9232

        /parent/directory/path/run3/
        /parent/directory/path/run3/events.out.tfevents.9232
  x = EventMultiplexer().AddRunsFromDirectory('/parent/directory/path')
  (which is equivalent to:)
  x = EventMultiplexer({'run1': '/parent/directory/path/run1', 'run2':...}
  ```

  If you would like to watch `/parent/directory/path`, wait for it to be created
    (if necessary) and then periodically pick up new runs, use
    `AutoloadingMultiplexer`

  @@__init__
  @@AddRun
  @@AddRunsFromDirectory
  @@Reload
  @@AutoUpdate
  @@Runs
  @@Scalars
  @@Graph
  @@Histograms
  @@CompressedHistograms
  @@Images
  """

  def __init__(self, run_path_map=None,
               size_guidance=event_accumulator.DEFAULT_SIZE_GUIDANCE):
    """Constructor for the `EventMultiplexer`.

    Args:
      run_path_map: Dict `{run: path}` which specifies the
        name of a run, and the path to find the associated events. If it is
        None, then the EventMultiplexer initializes without any runs.
      size_guidance: A dictionary mapping from `tagType` to the number of items
        to store for each tag of that type. See
        `event_ccumulator.EventAccumulator` for details.
    """
    self._accumulators_mutex = threading.Lock()
    self._accumulators = {}
    self._paths = {}
    self._reload_called = False
    self._autoupdate_called = False
    self._autoupdate_interval = None
    self._size_guidance = size_guidance
    if run_path_map is not None:
      for (run, path) in run_path_map.iteritems():
        self.AddRun(path, run)

  def AddRun(self, path, name=None):
    """Add a run to the multiplexer.

    If the name is not specified, it is the same as the path.

    If a run by that name exists, and we are already watching the right path,
      do nothing. If we are watching a different path, replace the event
      accumulator.

    If `AutoUpdate` or `Reload` have been called, it will `AutoUpdate` or
    `Reload` the newly created accumulators. This maintains the invariant that
    once the Multiplexer was activated, all of its accumulators are active.

    Args:
      path: Path to the event files (or event directory) for given run.
      name: Name of the run to add. If not provided, is set to path.

    Returns:
      The `EventMultiplexer`.
    """
    if name is None or name is '':
      name = path
    accumulator = None
    with self._accumulators_mutex:
      if name not in self._accumulators or self._paths[name] != path:
        if name in self._paths and self._paths[name] != path:
          # TODO(danmane) - Make it impossible to overwrite an old path with
          # a new path (just give the new path a distinct name)
          logging.warning('Conflict for name %s: old path %s, new path %s' %
                          (name, self._paths[name], path))
        logging.info('Constructing EventAccumulator for %s', path)
        accumulator = event_accumulator.EventAccumulator(path,
                                                         self._size_guidance)
        self._accumulators[name] = accumulator
        self._paths[name] = path
    if accumulator:
      if self._reload_called:
        accumulator.Reload()
      if self._autoupdate_called:
        accumulator.AutoUpdate(self._autoupdate_interval)
    return self

  def AddRunsFromDirectory(self, path, name=None):
    """Load runs from a directory, assuming each subdirectory is a run.

    If path doesn't exist, no-op. This ensures that it is safe to call
      `AddRunsFromDirectory` multiple times, even before the directory is made.

    If the directory contains TensorFlow event files, it is itself treated as a
      run.

    If the `EventMultiplexer` is already loaded or autoupdating, this will cause
    the newly created accumulators to also `Reload()` or `AutoUpdate()`.

    Args:
      path: A string path to a directory to load runs from.
      name: Optionally, what name to apply to the runs. If name is provided
        and the directory contains run subdirectories, the name of each subrun
        is the concatenation of the parent name and the subdirectory name. If
        name is provided and the directory contains event files, then a run
        is added called "name" and with the events from the path.

    Raises:
      ValueError: If the path exists and isn't a directory.

    Returns:
      The `EventMultiplexer`.
    """
    if not gfile.Exists(path):
      return  # Maybe it hasn't been created yet, fail silently to retry later
    if not gfile.IsDirectory(path):
      raise ValueError('Path exists and is not a directory, %s'  % path)
    paths = gfile.ListDirectory(path)
    is_directory = lambda x: gfile.IsDirectory(os.path.join(path, x))
    subdirectories = filter(is_directory, paths)
    for s in subdirectories:
      if name:
        subname = '/'.join([name, s])
      else:
        subname = s
      self.AddRun(os.path.join(path, s), subname)

    if filter(event_accumulator.IsTensorFlowEventsFile, paths):
      directory_name = os.path.split(path)[1]
      logging.info('Directory %s has event files; loading' % directory_name)
      if name:
        dname = name
      else:
        dname = directory_name
      self.AddRun(path, dname)
    return self

  def Reload(self):
    """Call `Reload` on every `EventAccumulator`."""
    self._reload_called = True
    with self._accumulators_mutex:
      loaders = self._accumulators.values()

    for l in loaders:
      l.Reload()
    return self

  def AutoUpdate(self, interval=60):
    """Call `AutoUpdate(interval)` on every `EventAccumulator`."""
    self._autoupdate_interval = interval
    self._autoupdate_called = True
    with self._accumulators_mutex:
      loaders = self._accumulators.values()
    for l in loaders:
      l.AutoUpdate(interval)
    return self

  def Scalars(self, run, tag):
    """Retrieve the scalar events associated with a run and tag.

    Args:
      run: A string name of the run for which values are retrieved.
      tag: A string name of the tag for which values are retrieved.

    Raises:
      KeyError: If the run is not found, or the tag is not available for
        the given run.
      RuntimeError: If the run's `EventAccumulator` has not been activated.

    Returns:
      An array of `event_accumulator.ScalarEvents`.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.Scalars(tag)

  def Graph(self, run):
    """Retrieve the graphs associated with the provided run.

    Args:
      run: A string name of a run to load the graph for.

    Raises:
      KeyError: If the run is not found.
      ValueError: If the run does not have an associated graph.
      RuntimeError: If the run's EventAccumulator has not been activated.

    Returns:
      The `graph_def` protobuf data structure.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.Graph()

  def Histograms(self, run, tag):
    """Retrieve the histogram events associated with a run and tag.

    Args:
      run: A string name of the run for which values are retrieved.
      tag: A string name of the tag for which values are retrieved.

    Raises:
      KeyError: If the run is not found, or the tag is not available for
        the given run.
      RuntimeError: If the run's `EventAccumulator` has not been activated.

    Returns:
      An array of `event_accumulator.HistogramEvents`.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.Histograms(tag)

  def CompressedHistograms(self, run, tag):
    """Retrieve the compressed histogram events associated with a run and tag.

    Args:
      run: A string name of the run for which values are retrieved.
      tag: A string name of the tag for which values are retrieved.

    Raises:
      KeyError: If the run is not found, or the tag is not available for
        the given run.
      RuntimeError: If the run's EventAccumulator has not been activated.

    Returns:
      An array of `event_accumulator.CompressedHistogramEvents`.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.CompressedHistograms(tag)

  def Images(self, run, tag):
    """Retrieve the image events associated with a run and tag.

    Args:
      run: A string name of the run for which values are retrieved.
      tag: A string name of the tag for which values are retrieved.

    Raises:
      KeyError: If the run is not found, or the tag is not available for
        the given run.
      RuntimeError: If the run's `EventAccumulator` has not been activated.

    Returns:
      An array of `event_accumulator.ImageEvents`.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.Images(tag)

  def Runs(self):
    """Return all the run names in the `EventMultiplexer`.

    Returns:
    ```
      {runName: { images: [tag1, tag2, tag3],
                  scalarValues: [tagA, tagB, tagC],
                  histograms: [tagX, tagY, tagZ],
                  compressedHistograms: [tagX, tagY, tagZ],
                  graph: true}}
    ```
    """
    with self._accumulators_mutex:
      # To avoid nested locks, we construct a copy of the run-accumulator map
      items = list(self._accumulators.iteritems())
    return {
        run_name: accumulator.Tags()
        for run_name, accumulator in items
    }

  def _GetAccumulator(self, run):
    with self._accumulators_mutex:
      return self._accumulators[run]


def AutoloadingMultiplexer(path_to_run, interval_secs=60,
    size_guidance=event_accumulator.DEFAULT_SIZE_GUIDANCE):
  """Create an `EventMultiplexer` that automatically loads runs in directories.

  Args:
    path_to_run: Dict `{path: name}` which specifies the path to a directory,
      and its name (or `None`). The path may contain tfevents files (in which
      case they are loaded, with name as the name of the run) and subdirectories
      containing tfevents files (in which case each subdirectory is added as a
      run, named `'name/subdirectory'`).

    interval_secs: How often to poll the directory for new runs.
    size_guidance: How much data to store for each tag of various types - see
      `event_accumulator.EventAccumulator`.

  Returns:
    The multiplexer which will automatically load from the directories.

  Raises:
    ValueError: if `path_to_run` is `None`
    TypeError: if `path_to_run` is not a dict
  """
  multiplexer = EventMultiplexer(size_guidance=size_guidance)
  if path_to_run is None:
    raise ValueError('Cant construct an autoloading multiplexer without runs.')
  if not isinstance(path_to_run, dict):
    raise TypeError('path_to_run should be a dict, was %s', path_to_run)
  def Load():
    for (path, name) in path_to_run.iteritems():
      logging.info('Checking for new runs in %s', path)
      multiplexer.AddRunsFromDirectory(path, name)
    t = threading.Timer(interval_secs, Load)
    t.daemon = True
    t.start()
  t = threading.Timer(0, Load)
  t.daemon = True
  t.start()
  return multiplexer
