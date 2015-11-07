"""Contains the implementation for the DirectoryWatcher class."""
import os

from tensorflow.python.platform import gfile
from tensorflow.python.platform import logging


class DirectoryWatcher(object):
  """A DirectoryWatcher wraps a loader to load from a directory.

  A loader reads a file on disk and produces some kind of values as an
  iterator. A DirectoryWatcher takes a directory with one file at a time being
  written to and a factory for loaders and watches all the files at once.

  This class is *only* valid under the assumption that files are never removed
  and the only file ever changed is whichever one is lexicographically last.
  """

  def __init__(self, directory, loader_factory, path_filter=lambda x: True):
    """Constructs a new DirectoryWatcher.

    Args:
      directory: The directory to watch. The directory doesn't have to exist.
      loader_factory: A factory for creating loaders. The factory should take a
        file path and return an object that has a Load method returning an
        iterator that will yield all events that have not been yielded yet.
      path_filter: Only files whose full path matches this predicate will be
        loaded. If not specified, all files are loaded.

    Raises:
      ValueError: If directory or loader_factory is None.
    """
    if directory is None:
      raise ValueError('A directory is required')
    if loader_factory is None:
      raise ValueError('A loader factory is required')
    self._directory = directory
    self._loader_factory = loader_factory
    self._loader = None
    self._path = None
    self._path_filter = path_filter

  def Load(self):
    """Loads new values from disk.

    The watcher will load from one file at a time; as soon as that file stops
    yielding events, it will move on to the next file. We assume that old files
    are never modified after a newer file has been written. As a result, Load()
    can be called multiple times in a row without losing events that have not
    been yielded yet. In other words, we guarantee that every event will be
    yielded exactly once.

    Yields:
      All values that were written to disk that have not been yielded yet.
    """

    # If the loader exists, check it for a value.
    if not self._loader:
      self._InitializeLoader()

    while True:
      # Yield all the new events in the file we're currently loading from.
      for event in self._loader.Load():
        yield event

      next_path = self._GetNextPath()
      if not next_path:
        logging.info('No more files in %s', self._directory)
        # Current file is empty and there are no new files, so we're done.
        return

      # There's a new file, so check to make sure there weren't any events
      # written between when we finished reading the current file and when we
      # checked for the new one. The sequence of events might look something
      # like this:
      #
      # 1. Event #1 written to file #1.
      # 2. We check for events and yield event #1 from file #1
      # 3. We check for events and see that there are no more events in file #1.
      # 4. Event #2 is written to file #1.
      # 5. Event #3 is written to file #2.
      # 6. We check for a new file and see that file #2 exists.
      #
      # Without this loop, we would miss event #2. We're also guaranteed by the
      # loader contract that no more events will be written to file #1 after
      # events start being written to file #2, so we don't have to worry about
      # that.
      for event in self._loader.Load():
        yield event

      logging.info('Directory watcher for %s advancing to file %s',
                   self._directory, next_path)

      # Advance to the next file and start over.
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
    """Returns the path of the next file to use or None if no file exists."""
    sorted_paths = [os.path.join(self._directory, path)
                    for path in sorted(gfile.ListDirectory(self._directory))]
    # We filter here so the filter gets the full directory name.
    filtered_paths = (path for path in sorted_paths
                      if self._path_filter(path) and path > self._path)
    return next(filtered_paths, None)
