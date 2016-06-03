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

import os
import socket

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import status_bar
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_file_inspector as efi
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import server

flags.DEFINE_string('logdir', None, """logdir specifies the directory where
TensorBoard will look to find TensorFlow event files that it can display.
TensorBoard will recursively walk the directory structure rooted at logdir,
looking for .*tfevents.* files.

You may also pass a comma separated list of log directories, and TensorBoard
will watch each directory. You can also assign names to individual log
directories by putting a colon between the name and the path, as in

tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
""")

flags.DEFINE_boolean('debug', False, 'Whether to run the app in debug mode. '
                     'This increases log verbosity to DEBUG.')

flags.DEFINE_string('host', '0.0.0.0', 'What host to listen to. Defaults to '
                    'serving on 0.0.0.0, set to 127.0.0.1 (localhost) to'
                    'disable remote access (also quiets security warnings).')

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

flags.DEFINE_integer('port', 6006, 'What port to serve TensorBoard on.')

flags.DEFINE_boolean('purge_orphaned_data', True, 'Whether to purge data that '
                     'may have been orphaned due to TensorBoard restarts. '
                     'Disabling purge_orphaned_data can be used to debug data '
                     'disappearance.')

flags.DEFINE_integer('reload_interval', 60, 'How often the backend should load '
                     'more data.')

FLAGS = flags.FLAGS


def main(unused_argv=None):
  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)
    logging.info('TensorBoard is in debug mode.')

  if FLAGS.inspect:
    logging.info('Not bringing up TensorBoard, but inspecting event files.')
    efi.inspect(logdir=FLAGS.logdir,
                event_file=FLAGS.event_file,
                tag=FLAGS.tag)
    return 0

  if not FLAGS.logdir:
    msg = ('A logdir must be specified. Run `tensorboard --help` for '
           'details and examples.')
    logging.error(msg)
    print(msg)
    return -1

  logging.info('Starting TensorBoard in directory %s', os.getcwd())
  path_to_run = server.ParseEventFilesSpec(FLAGS.logdir)
  logging.info('TensorBoard path_to_run is: %s', path_to_run)

  multiplexer = event_multiplexer.EventMultiplexer(
      size_guidance=server.TENSORBOARD_SIZE_GUIDANCE,
      purge_orphaned_data=FLAGS.purge_orphaned_data)
  server.StartMultiplexerReloadingThread(multiplexer, path_to_run,
                                         FLAGS.reload_interval)
  try:
    tb_server = server.BuildServer(multiplexer, FLAGS.host, FLAGS.port)
  except socket.error:
    if FLAGS.port == 0:
      msg = 'Unable to find any open ports.'
      logging.error(msg)
      print(msg)
      return -2
    else:
      msg = 'Tried to connect to port %d, but address is in use.' % FLAGS.port
      logging.error(msg)
      print(msg)
      return -3

  try:
    tag = resource_loader.load_resource('tensorboard/TAG').strip()
    logging.info('TensorBoard is tag: %s', tag)
  except IOError:
    logging.warning('Unable to read TensorBoard tag')
    tag = ''

  status_bar.SetupStatusBarInsideGoogle('TensorBoard %s' % tag, FLAGS.port)
  print('Starting TensorBoard %s on port %d' % (tag, FLAGS.port))
  print('(You can navigate to http://%s:%d)' % (FLAGS.host, FLAGS.port))
  tb_server.serve_forever()


if __name__ == '__main__':
  app.run()
