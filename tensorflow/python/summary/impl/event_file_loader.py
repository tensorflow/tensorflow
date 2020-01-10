"""Functionality for loading events from a record file."""

from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import logging


class EventFileLoader(object):
  """An EventLoader is an iterator that yields Event protos."""

  def __init__(self, file_path):
    if file_path is None:
      raise ValueError('A file path is required')
    logging.debug('Opening a record reader pointing at %s', file_path)
    self._reader = pywrap_tensorflow.PyRecordReader_New(file_path, 0)
    # Store it for logging purposes.
    self._file_path = file_path
    if not self._reader:
      raise IOError('Failed to open a record reader pointing to %s' % file_path)

  def Load(self):
    """Loads all new values from disk.

    Calling Load multiple times in a row will not 'drop' events as long as the
    return value is not iterated over.

    Yields:
      All values that were written to disk that have not been yielded yet.
    """
    while self._reader.GetNext():
      logging.debug('Got an event from %s', self._file_path)
      event = event_pb2.Event()
      event.ParseFromString(self._reader.record())
      yield event
    logging.debug('No more events in %s', self._file_path)


def main(argv):
  if len(argv) != 2:
    print 'Usage: event_file_loader <path-to-the-recordio-file>'
    return 1
  loader = EventFileLoader(argv[1])
  for event in loader.Load():
    print event


if __name__ == '__main__':
  app.run()
