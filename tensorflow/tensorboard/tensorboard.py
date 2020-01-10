"""Serve TensorFlow summary data to a web frontend.

This is a simple web server to proxy data from the event_loader to the web, and
serve static web files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import BaseHTTPServer
import functools
import os
import socket
import SocketServer

import tensorflow.python.platform

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import logging
from tensorflow.python.platform import status_bar
from tensorflow.python.summary import event_accumulator
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard import tensorboard_handler

flags.DEFINE_string('logdir', None, """logdir specifies the directory where
TensorBoard will look to find TensorFlow event files that it can display.
TensorBoard will recursively walk the directory structure rooted at logdir,
looking for .*tfevents.* files.

You may also pass a comma seperated list of log directories, and TensorBoard
will watch each directory. You can also assign names to individual log
directories by putting a colon between the name and the path, as in

tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
""")

flags.DEFINE_boolean('debug', False, 'Whether to run the app in debug mode. '
                     'This increases log verbosity to DEBUG.')


flags.DEFINE_string('host', '127.0.0.1', 'What host to listen to. Defaults to '
                    'serving on localhost, set to 0.0.0.0 for remote access.')

flags.DEFINE_integer('port', 6006, 'What port to serve TensorBoard on.')

FLAGS = flags.FLAGS

# How many elements to store per tag, by tag type
TENSORBOARD_SIZE_GUIDANCE = {
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.SCALARS: 10000,
    event_accumulator.HISTOGRAMS: 1,
}


def ParseEventFilesFlag(flag_value):
  """Parses the logdir flag into a map from paths to run group names.

  The events files flag format is a comma-separated list of path specifications.
  A path specification either looks like 'group_name:/path/to/directory' or
  '/path/to/directory'; in the latter case, the group is unnamed. Group names
  cannot start with a forward slash: /foo:bar/baz will be interpreted as a
  spec with no name and path '/foo:bar/baz'.

  Globs are not supported.

  Args:
    flag_value: A comma-separated list of run specifications.
  Returns:
    A dict mapping directory paths to names like {'/path/to/directory': 'name'}.
    Groups without an explicit name are named after their path. If flag_value
    is None, returns an empty dict, which is helpful for testing things that
    don't require any valid runs.
  """
  files = {}
  if flag_value is None:
    return files
  for specification in flag_value.split(','):
    # If the spec looks like /foo:bar/baz, then we assume it's a path with a
    # colon.
    if ':' in specification and specification[0] != '/':
      # We split at most once so run_name:/path:with/a/colon will work.
      run_name, path = specification.split(':', 1)
    else:
      run_name = None
      path = specification
    files[path] = run_name
  return files


class ThreadedHTTPServer(SocketServer.ThreadingMixIn,
                         BaseHTTPServer.HTTPServer):
  """A threaded HTTP server."""
  daemon = True


def main(unused_argv=None):
  # Change current working directory to tensorflow/'s parent directory.
  server_root = os.path.join(os.path.dirname(__file__),
                             os.pardir, os.pardir)
  os.chdir(server_root)

  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)

  if not FLAGS.logdir:
    logging.error('A logdir must be specified. Run `tensorboard --help` for '
                  'details and examples.')
    return -1

  if FLAGS.debug:
    logging.info('Starting TensorBoard in directory %s', os.getcwd())

  path_to_run = ParseEventFilesFlag(FLAGS.logdir)
  multiplexer = event_multiplexer.AutoloadingMultiplexer(
      path_to_run=path_to_run, interval_secs=60,
      size_guidance=TENSORBOARD_SIZE_GUIDANCE)

  multiplexer.AutoUpdate(interval=30)

  factory = functools.partial(tensorboard_handler.TensorboardHandler,
                              multiplexer)
  try:
    server = ThreadedHTTPServer((FLAGS.host, FLAGS.port), factory)
  except socket.error:
    logging.error('Tried to connect to port %d, but that address is in use.',
                  FLAGS.port)
    return -2

  status_bar.SetupStatusBarInsideGoogle('TensorBoard', FLAGS.port)
  print('Starting TensorBoard on port %d' % FLAGS.port)
  print('(You can navigate to http://localhost:%d)' % FLAGS.port)
  server.serve_forever()


if __name__ == '__main__':
  app.run()
