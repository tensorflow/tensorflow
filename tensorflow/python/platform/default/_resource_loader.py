"""Read a file and return its contents."""

import os.path

from tensorflow.python.platform import logging


def load_resource(path):
  """Load the resource at given path, where path is relative to tensorflow/.

  Args:
    path: a string resource path relative to tensorflow/.

  Returns:
    The contents of that resource.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  path = os.path.join('tensorflow', path)
  path = os.path.abspath(path)
  try:
    with open(path, 'rb') as f:
      return f.read()
  except IOError as e:
    logging.warning('IOError %s on path %s' % (e, path))
