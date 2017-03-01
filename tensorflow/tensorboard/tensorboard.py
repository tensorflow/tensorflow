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
"""Serve TensorFlow summary data to a web frontend.

This is a simple web server to proxy data from the event_loader to the web, and
serve static web files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as base_logging
import os
import socket
import sys
from werkzeug import serving

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_file_inspector as efi
from tensorflow.tensorboard.backend import application


# TensorBoard flags

flags.DEFINE_string('logdir', '', """logdir specifies the directory where
TensorBoard will look to find TensorFlow event files that it can display.
TensorBoard will recursively walk the directory structure rooted at logdir,
looking for .*tfevents.* files.

You may also pass a comma separated list of log directories, and TensorBoard
will watch each directory. You can also assign names to individual log
directories by putting a colon between the name and the path, as in

tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
""")

flags.DEFINE_string('host', '0.0.0.0', 'What host to listen to. Defaults to '
                    'serving on 0.0.0.0, set to 127.0.0.1 (localhost) to'
                    'disable remote access (also quiets security warnings).')

flags.DEFINE_integer('port', 6006, 'What port to serve TensorBoard on.')

flags.DEFINE_boolean('purge_orphaned_data', True, 'Whether to purge data that '
                     'may have been orphaned due to TensorBoard restarts. '
                     'Disabling purge_orphaned_data can be used to debug data '
                     'disappearance.')

flags.DEFINE_integer('reload_interval', 5, 'How often the backend should load '
                     'more data.')

# Inspect Mode flags

flags.DEFINE_boolean('inspect', False, """Use this flag to print out a digest
of your event files to the command line, when no data is shown on TensorBoard or
the data shown looks weird.

Example usages:
tensorboard --inspect --event_file=myevents.out
tensorboard --inspect --event_file=myevents.out --tag=loss
tensorboard --inspect --logdir=mylogdir
tensorboard --inspect --logdir=mylogdir --tag=loss

See tensorflow/python/summary/event_file_inspector.py for more info and
detailed usage.
""")
flags.DEFINE_string(
    'tag', '',
    'The particular tag to query for. Only used if --inspect is present')
flags.DEFINE_string(
    'event_file', '',
    'The particular event file to query for. Only used if --inspect is present '
    'and --logdir is not specified.')

FLAGS = flags.FLAGS


def create_tb_app():
  """Read the flags, and create a TensorBoard WSGI application."""
  if not FLAGS.logdir:
    raise ValueError('A logdir must be specified. Run `tensorboard --help` for '
                     'details and examples.')

  logdir = os.path.expanduser(FLAGS.logdir)
  return application.standard_tensorboard_wsgi(
      logdir=logdir,
      purge_orphaned_data=FLAGS.purge_orphaned_data,
      reload_interval=FLAGS.reload_interval)


def run_simple_server(tb_app):
  """Start serving TensorBoard, and print some messages to console."""
  # Mute the werkzeug logging.
  base_logging.getLogger('werkzeug').setLevel(base_logging.WARNING)

  try:
    server = serving.make_server(FLAGS.host, FLAGS.port, tb_app, threaded=True)
    server.daemon_threads = True
  except socket.error:
    if FLAGS.port == 0:
      msg = 'TensorBoard unable to find any open port'
    else:
      msg = (
          'TensorBoard attempted to bind to port %d, but it was already in use'
          % FLAGS.port)
    logging.error(msg)
    print(msg)
    exit(-1)

  port = server.socket.getsockname()[1]
  msg = 'Starting TensorBoard %s at http://%s:%d' % (tb_app.tag, FLAGS.host,
                                                     port)
  print(msg)
  logging.info(msg)
  print('(Press CTRL+C to quit)')
  sys.stdout.flush()

  server.serve_forever()


def main(unused_argv=None):
  if FLAGS.inspect:
    logging.info('Not bringing up TensorBoard, but inspecting event files.')
    event_file = os.path.expanduser(FLAGS.event_file)
    efi.inspect(FLAGS.logdir, event_file, FLAGS.tag)
    return 0
  else:
    tb = create_tb_app()
    run_simple_server(tb)

if __name__ == '__main__':
  app.run()
