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

"""Contains the implementation for the DirectoryWatcher class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect

from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary.impl import io_wrapper


class DirectoryWatcher(object):
  """A DirectoryWatcher wraps a loader to load from a sequence of paths.

  A loader reads a path and produces some kind of values as an iterator. A
  DirectoryWatcher takes a directory, a factory for loaders, and optionally a
  path filter and watches all the paths inside that directory.

  This class is only valid under the assumption that only one path will be
  written to by the data source at a time and that once the source stops writing
  to a path, it will start writing to a new path that's lexicographically
  greater and never come back. It uses some heuristics to check whether this is
  true based on tracking changes to the files' sizes, but the check can have
  false negatives. However, it should have no false positives.
  """

  def __init__(self, directory, loader_factory, path_filter=lambda x: True):
    """Constructs a new DirectoryWatcher.

    Args:
      directory: The directory to load files from.
      loader_factory: A factory for creating loaders. The factory should take a
        path and return an object that has a Load method returning an
        iterator that will yield all events that have not been yielded yet.
      path_filter: If specified, only paths matching this filter are loaded.

    Raises:
      ValueError: If path_provider or loader_factory are None.
    """
    if directory is None:
      raise ValueError('A directory is required')
    if loader_factory is None:
      raise ValueError('A loader factory is required')
    self._directory = directory
    self._path = None
    self._loader_factory = loader_factory
    self._loader = None
    self._path_filter = path_filter
    self._ooo_writes_detected = False
    # The file size for each file at the time it was finalized.
    self._finalized_sizes = {}

  def Load(self):
    """Loads new values.

    The watcher will load from one path at a time; as soon as that path stops
    yielding events, it will move on to the next path. We assume that old paths
    are never modified after a newer path has been written. As a result, Load()
    can be called multiple times in a row without losing events that have not
    been yielded yet. In other words, we guarantee that every event will be
    yielded exactly once.

    Yields:
      All values that have not been yielded yet.

    Raises:
      DirectoryDeletedError: If the directory has been permanently deleted
        (as opposed to being temporarily unavailable).
    """
    try:
      for event in self._LoadInternal():
        yield event
    except errors.OpError:
      if not gfile.Exists(self._directory):
        raise DirectoryDeletedError(
            'Directory %s has been permanently deleted' % self._directory)

  def _LoadInternal(self):
    """Internal implementation of Load().

    The only difference between this and Load() is that the latter will throw
    DirectoryDeletedError on I/O errors if it thinks that the directory has been
    permanently deleted.

    Yields:
      All values that have not been yielded yet.
    """

    # If the loader exists, check it for a value.
    if not self._loader:
      self._InitializeLoader()

    while True:
      # Yield all the new events in the path we're currently loading from.
      for event in self._loader.Load():
        yield event

      next_path = self._GetNextPath()
      if not next_path:
        logging.info('No path found after %s', self._path)
        # Current path is empty and there are no new paths, so we're done.
        return

      # There's a new path, so check to make sure there weren't any events
      # written between when we finished reading the current path and when we
      # checked for the new one. The sequence of events might look something
      # like this:
      #
      # 1. Event #1 written to path #1.
      # 2. We check for events and yield event #1 from path #1
      # 3. We check for events and see that there are no more events in path #1.
      # 4. Event #2 is written to path #1.
      # 5. Event #3 is written to path #2.
      # 6. We check for a new path and see that path #2 exists.
      #
      # Without this loop, we would miss event #2. We're also guaranteed by the
      # loader contract that no more events will be written to path #1 after
      # events start being written to path #2, so we don't have to worry about
      # that.
      for event in self._loader.Load():
        yield event

      logging.info('Directory watcher advancing from %s to %s', self._path,
                   next_path)

      # Advance to the next path and start over.
      self._SetPath(next_path)

  # The number of paths before the current one to check for out of order writes.
  _OOO_WRITE_CHECK_COUNT = 20

  def OutOfOrderWritesDetected(self):
    """Returns whether any out-of-order writes have been detected.

    Out-of-order writes are only checked as part of the Load() iterator. Once an
    out-of-order write is detected, this function will always return true.

    Note that out-of-order write detection is not performed on GCS paths, so
    this function will always return false.

    Returns:
      Whether any out-of-order write has ever been detected by this watcher.

    """
    return self._ooo_writes_detected

  def _InitializeLoader(self):
    path = self._GetNextPath()
    if path:
      self._SetPath(path)
    else:
      raise StopIteration

  def _SetPath(self, path):
    """Sets the current path to watch for new events.

    This also records the size of the old path, if any. If the size can't be
    found, an error is logged.

    Args:
      path: The full path of the file to watch.
    """
    old_path = self._path
    if old_path and not io_wrapper.IsGCSPath(old_path):
      try:
        # We're done with the path, so store its size.
        size = gfile.Stat(old_path).length
        logging.debug('Setting latest size of %s to %d', old_path, size)
        self._finalized_sizes[old_path] = size
      except errors.OpError as e:
        logging.error('Unable to get size of %s: %s', old_path, e)

    self._path = path
    self._loader = self._loader_factory(path)

  def _GetNextPath(self):
    """Gets the next path to load from.

    This function also does the checking for out-of-order writes as it iterates
    through the paths.

    Returns:
      The next path to load events from, or None if there are no more paths.
    """
    paths = sorted(path
                   for path in io_wrapper.ListDirectoryAbsolute(self._directory)
                   if self._path_filter(path))
    if not paths:
      return None

    if self._path is None:
      return paths[0]

    # Don't bother checking if the paths are GCS (which we can't check) or if
    # we've already detected an OOO write.
    if not io_wrapper.IsGCSPath(paths[0]) and not self._ooo_writes_detected:
      # Check the previous _OOO_WRITE_CHECK_COUNT paths for out of order writes.
      current_path_index = bisect.bisect_left(paths, self._path)
      ooo_check_start = max(0, current_path_index - self._OOO_WRITE_CHECK_COUNT)
      for path in paths[ooo_check_start:current_path_index]:
        if self._HasOOOWrite(path):
          self._ooo_writes_detected = True
          break

    next_paths = list(path
                      for path in paths
                      if self._path is None or path > self._path)
    if next_paths:
      return min(next_paths)
    else:
      return None

  def _HasOOOWrite(self, path):
    """Returns whether the path has had an out-of-order write."""
    # Check the sizes of each path before the current one.
    size = gfile.Stat(path).length
    old_size = self._finalized_sizes.get(path, None)
    if size != old_size:
      if old_size is None:
        logging.error('File %s created after file %s even though it\'s '
                      'lexicographically earlier', path, self._path)
      else:
        logging.error('File %s updated even though the current file is %s',
                      path, self._path)
      return True
    else:
      return False


class DirectoryDeletedError(Exception):
  """Thrown by Load() when the directory is *permanently* gone.

  We distinguish this from temporary errors so that other code can decide to
  drop all of our data only when a directory has been intentionally deleted,
  as opposed to due to transient filesystem errors.
  """
  pass
