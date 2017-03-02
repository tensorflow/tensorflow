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
"""Provides an interface for working with multiple event files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading

import six

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_accumulator
from tensorflow.python.summary.impl import directory_watcher
from tensorflow.python.summary.impl import io_wrapper


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
  @@Tensors
  """

  def __init__(self,
               run_path_map=None,
               size_guidance=event_accumulator.DEFAULT_SIZE_GUIDANCE,
               purge_orphaned_data=True):
    """Constructor for the `EventMultiplexer`.

    Args:
      run_path_map: Dict `{run: path}` which specifies the
        name of a run, and the path to find the associated events. If it is
        None, then the EventMultiplexer initializes without any runs.
      size_guidance: A dictionary mapping from `tagType` to the number of items
        to store for each tag of that type. See
        `event_accumulator.EventAccumulator` for details.
      purge_orphaned_data: Whether to discard any events that were "orphaned" by
        a TensorFlow restart.
    """
    logging.info('Event Multiplexer initializing.')
    self._accumulators_mutex = threading.Lock()
    self._accumulators = {}
    self._paths = {}
    self._reload_called = False
    self._size_guidance = size_guidance
    self.purge_orphaned_data = purge_orphaned_data
    if run_path_map is not None:
      logging.info('Event Multplexer doing initialization load for %s',
                   run_path_map)
      for (run, path) in six.iteritems(run_path_map):
        self.AddRun(path, run)
    logging.info('Event Multiplexer done initializing')

  def AddRun(self, path, name=None):
    """Add a run to the multiplexer.

    If the name is not specified, it is the same as the path.

    If a run by that name exists, and we are already watching the right path,
      do nothing. If we are watching a different path, replace the event
      accumulator.

    If `Reload` has been called, it will `Reload` the newly created
    accumulators.

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
          logging.warning('Conflict for name %s: old path %s, new path %s',
                          name, self._paths[name], path)
        logging.info('Constructing EventAccumulator for %s', path)
        accumulator = event_accumulator.EventAccumulator(
            path,
            size_guidance=self._size_guidance,
            purge_orphaned_data=self.purge_orphaned_data)
        self._accumulators[name] = accumulator
        self._paths[name] = path
    if accumulator:
      if self._reload_called:
        accumulator.Reload()
    return self

  def AddRunsFromDirectory(self, path, name=None):
    """Load runs from a directory; recursively walks subdirectories.

    If path doesn't exist, no-op. This ensures that it is safe to call
      `AddRunsFromDirectory` multiple times, even before the directory is made.

    If path is a directory, load event files in the directory (if any exist) and
      recursively call AddRunsFromDirectory on any subdirectories. This mean you
      can call AddRunsFromDirectory at the root of a tree of event logs and
      TensorBoard will load them all.

    If the `EventMultiplexer` is already loaded this will cause
    the newly created accumulators to `Reload()`.
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
    logging.info('Starting AddRunsFromDirectory: %s', path)
    for subdir in GetLogdirSubdirectories(path):
      logging.info('Adding events from directory %s', subdir)
      rpath = os.path.relpath(subdir, path)
      subname = os.path.join(name, rpath) if name else rpath
      self.AddRun(subdir, name=subname)
    logging.info('Done with AddRunsFromDirectory: %s', path)
    return self

  def Reload(self):
    """Call `Reload` on every `EventAccumulator`."""
    logging.info('Beginning EventMultiplexer.Reload()')
    self._reload_called = True
    # Build a list so we're safe even if the list of accumulators is modified
    # even while we're reloading.
    with self._accumulators_mutex:
      items = list(self._accumulators.items())

    names_to_delete = set()
    for name, accumulator in items:
      try:
        accumulator.Reload()
      except (OSError, IOError) as e:
        logging.error("Unable to reload accumulator '%s': %s", name, e)
      except directory_watcher.DirectoryDeletedError:
        names_to_delete.add(name)

    with self._accumulators_mutex:
      for name in names_to_delete:
        logging.warning("Deleting accumulator '%s'", name)
        del self._accumulators[name]
    logging.info('Finished with EventMultiplexer.Reload()')
    return self

  def FirstEventTimestamp(self, run):
    """Return the timestamp of the first event of the given run.

    This may perform I/O if no events have been loaded yet for the run.

    Args:
      run: A string name of the run for which the timestamp is retrieved.

    Returns:
      The wall_time of the first event of the run, which will typically be
      seconds since the epoch.

    Raises:
      KeyError: If the run is not found.
      ValueError: If the run has no events loaded and there are no events on
        disk to load.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.FirstEventTimestamp()

  def Scalars(self, run, tag):
    """Retrieve the scalar events associated with a run and tag.

    Args:
      run: A string name of the run for which values are retrieved.
      tag: A string name of the tag for which values are retrieved.

    Raises:
      KeyError: If the run is not found, or the tag is not available for
        the given run.

    Returns:
      An array of `event_accumulator.ScalarEvents`.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.Scalars(tag)

  def HealthPills(self, run, node_name):
    """Retrieve the health pill events associated with a run and node name.

    Args:
      run: A string name of the run for which health pills are retrieved.
      node_name: A string name of the node for which health pills are retrieved.

    Raises:
      KeyError: If the run is not found, or the node name is not available for
        the given run.

    Returns:
      An array of `event_accumulator.HealthPillEvents`.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.HealthPills(node_name)

  def Graph(self, run):
    """Retrieve the graph associated with the provided run.

    Args:
      run: A string name of a run to load the graph for.

    Raises:
      KeyError: If the run is not found.
      ValueError: If the run does not have an associated graph.

    Returns:
      The `GraphDef` protobuf data structure.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.Graph()

  def MetaGraph(self, run):
    """Retrieve the metagraph associated with the provided run.

    Args:
      run: A string name of a run to load the graph for.

    Raises:
      KeyError: If the run is not found.
      ValueError: If the run does not have an associated graph.

    Returns:
      The `MetaGraphDef` protobuf data structure.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.MetaGraph()

  def RunMetadata(self, run, tag):
    """Get the session.run() metadata associated with a TensorFlow run and tag.

    Args:
      run: A string name of a TensorFlow run.
      tag: A string name of the tag associated with a particular session.run().

    Raises:
      KeyError: If the run is not found, or the tag is not available for the
        given run.

    Returns:
      The metadata in the form of `RunMetadata` protobuf data structure.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.RunMetadata(tag)

  def Histograms(self, run, tag):
    """Retrieve the histogram events associated with a run and tag.

    Args:
      run: A string name of the run for which values are retrieved.
      tag: A string name of the tag for which values are retrieved.

    Raises:
      KeyError: If the run is not found, or the tag is not available for
        the given run.

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

    Returns:
      An array of `event_accumulator.ImageEvents`.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.Images(tag)

  def Audio(self, run, tag):
    """Retrieve the audio events associated with a run and tag.

    Args:
      run: A string name of the run for which values are retrieved.
      tag: A string name of the tag for which values are retrieved.

    Raises:
      KeyError: If the run is not found, or the tag is not available for
        the given run.

    Returns:
      An array of `event_accumulator.AudioEvents`.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.Audio(tag)

  def Tensors(self, run, tag):
    """Retrieve the tensor events associated with a run and tag.

    Args:
      run: A string name of the run for which values are retrieved.
      tag: A string name of the tag for which values are retrieved.

    Raises:
      KeyError: If the run is not found, or the tag is not available for
        the given run.

    Returns:
      An array of `event_accumulator.TensorEvent`s.
    """
    accumulator = self._GetAccumulator(run)
    return accumulator.Tensors(tag)

  def Runs(self):
    """Return all the run names in the `EventMultiplexer`.

    Returns:
    ```
      {runName: { images: [tag1, tag2, tag3],
                  scalarValues: [tagA, tagB, tagC],
                  histograms: [tagX, tagY, tagZ],
                  compressedHistograms: [tagX, tagY, tagZ],
                  graph: true, meta_graph: true}}
    ```
    """
    with self._accumulators_mutex:
      # To avoid nested locks, we construct a copy of the run-accumulator map
      items = list(six.iteritems(self._accumulators))
    return {run_name: accumulator.Tags() for run_name, accumulator in items}

  def RunPaths(self):
    """Returns a dict mapping run names to event file paths."""
    return self._paths

  def _GetAccumulator(self, run):
    with self._accumulators_mutex:
      return self._accumulators[run]


def GetLogdirSubdirectories(path):
  """Returns subdirectories with event files on path."""
  if gfile.Exists(path) and not gfile.IsDirectory(path):
    raise ValueError('GetLogdirSubdirectories: path exists and is not a '
                     'directory, %s' % path)

  # ListRecursively just yields nothing if the path doesn't exist.
  return (
      subdir
      for (subdir, files) in io_wrapper.ListRecursively(path)
      if list(filter(event_accumulator.IsTensorFlowEventsFile, files))
  )
