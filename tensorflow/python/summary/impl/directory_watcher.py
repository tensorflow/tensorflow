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

"""Contains the implementation for the DirectoryWatcher class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.platform import gfile
from tensorflow.python.platform import logging


class DirectoryWatcher(object):
  """A DirectoryWatcher wraps a loader to load from a sequence of paths.

  A loader reads a path and produces some kind of values as an iterator. A
  DirectoryWatcher takes a directory, a path provider (see below) to call to
  find new paths to load from, and a factory for loaders and watches all the
  paths inside that directory.

  A path provider is a function that, given either a path or None, returns the
  next path to load from (or None if there is no such path). This class is only
  valid under the assumption that only one path will be written to by the data
  source at a time, and that the path_provider will return the oldest data
  source that contains fresh data.

  """

  def __init__(self, path_provider, loader_factory):
    """Constructs a new DirectoryWatcher.

    Args:
      path_provider: The callback to invoke when trying to find a new path to
        load from. See the class documentation for the semantics of a path
        provider.
      loader_factory: A factory for creating loaders. The factory should take a
        path and return an object that has a Load method returning an
        iterator that will yield all events that have not been yielded yet.

    Raises:
      ValueError: If path_provider or loader_factory are None.
    """
    if path_provider is None:
      raise ValueError('A path provider is required')
    if loader_factory is None:
      raise ValueError('A loader factory is required')
    self._path_provider = path_provider
    self._path = None
    self._loader_factory = loader_factory
    self._loader = None

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

  def _InitializeLoader(self):
    path = self._GetNextPath()
    if path:
      self._SetPath(path)
    else:
      raise StopIteration

  def _SetPath(self, path):
    self._path = path
    self._loader = self._loader_factory(path)

  def _GetNextPath(self):
    """Returns the next path to use or None if no such path exists."""
    return self._path_provider(self._path)


def _SequentialProvider(path_source):
  """A provider that iterates over the output of a function that produces paths.

  _SequentialProvider takes in a path_source, which is a function that returns a
  list of all currently available paths. _SequentialProvider returns in a path
  provider (see documentation for the |DirectoryWatcher| class for the
  semantics) that will return the alphabetically next path after the current one
  (or the earliest path if the current path is None).

  The provider will never return a path which is alphanumerically less than the
  current path; as such, if the path source provides a high path (e.g. "c") and
  later doubles back and provides a low path (e.g. "b"), once the current path
  was set to "c" the _SequentialProvider will ignore the "b" and never return
  it.

  Args:
    path_source: A function that returns an iterable of paths.

  Returns:
    A path provider for use with DirectoryWatcher.

  """
  def _Provider(current_path):
    next_paths = list(path
                      for path in path_source()
                      if current_path is None or path > current_path)
    if next_paths:
      return min(next_paths)
    else:
      return None

  return _Provider


def SequentialGFileProvider(directory, path_filter=lambda x: True):
  """Provides the files in a directory that match the given filter."""
  def _Source():
    paths = (os.path.join(directory, path)
             for path in gfile.ListDirectory(directory))
    return (path for path in paths if path_filter(path))

  return _SequentialProvider(_Source)
