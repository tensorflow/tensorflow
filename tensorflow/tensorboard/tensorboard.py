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

import argparse
import os
import socket

from tensorflow.python.platform import app
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import status_bar
from tensorflow.python.platform import (
    tf_logging as logging)
from tensorflow.python.summary import (
    event_file_inspector as efi)
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import server

FLAGS = None


def main(unused_argv=None):
  logdir = os.path.expanduser(FLAGS.logdir)
  event_file = os.path.expanduser(FLAGS.event_file)

  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)
    logging.info('TensorBoard is in debug mode.')

  if FLAGS.inspect:
    logging.info('Not bringing up TensorBoard, but inspecting event files.')
    efi.inspect(logdir, event_file, FLAGS.tag)
    return 0

  if not logdir:
    msg = ('A logdir must be specified. Run `tensorboard --help` for '
           'details and examples.')
    logging.error(msg)
    print(msg)
    return -1

  logging.info('Starting TensorBoard in directory %s', os.getcwd())
  path_to_run = server.ParseEventFilesSpec(logdir)
  logging.info('TensorBoard path_to_run is: %s', path_to_run)

  multiplexer = event_multiplexer.EventMultiplexer(
      size_guidance=server.TENSORBOARD_SIZE_GUIDANCE,
      purge_orphaned_data=FLAGS.purge_orphaned_data)
  server.StartMultiplexerReloadingThread(multiplexer, path_to_run,
                                         FLAGS.reload_interval)
  try:
    tb_server = server.BuildServer(multiplexer, FLAGS.host, FLAGS.port, logdir)
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
    logging.info('Unable to read TensorBoard tag')
    tag = ''

  status_bar.SetupStatusBarInsideGoogle('TensorBoard %s' % tag, FLAGS.port)
  print('Starting TensorBoard %s on port %d' % (tag, FLAGS.port))

  if FLAGS.host == "0.0.0.0":
    try:
      host = socket.gethostbyname(socket.gethostname())
      print('(You can navigate to http://%s:%d)' % (host, FLAGS.port))
    except socket.gaierror:
      pass
  else:
    print('(You can navigate to http://%s:%d)' % (FLAGS.host, FLAGS.port))

  tb_server.serve_forever()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--logdir',
      type=str,
      default='',
      help="""\
      logdir specifies the directory where TensorBoard will look to find
      TensorFlow event files that it can display. TensorBoard will recursively
      walk the directory structure rooted at logdir, looking for .*tfevents.*
      files.

      You may also pass a comma separated list of log directories, and
      TensorBoard will watch each directory. You can also assign names to
      individual log directories by putting a colon between the name and the
      path, as in

      tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2\
      """
  )
  parser.add_argument(
      '--debug',
      default=False,
      help="""\
      Whether to run the app in debug mode. This increases log verbosity to
      DEBUG.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--host',
      type=str,
      default='0.0.0.0',
      help="""\
      What host to listen to. Defaults to serving on 0.0.0.0, set to 127.0.0.1
      (localhost) todisable remote access (also quiets security warnings).\
      """
  )
  parser.add_argument(
      '--inspect',
      default=False,
      help="""\
      Use this flag to print out a digest of your event files to the command
      line, when no data is shown on TensorBoard or the data shown looks weird.

      Example usages:
      tensorboard --inspect --event_file=myevents.out
      tensorboard --inspect --event_file=myevents.out --tag=loss
      tensorboard --inspect --logdir=mylogdir
      tensorboard --inspect --logdir=mylogdir --tag=loss

      See tensorflow/python/summary/event_file_inspector.py for
      more info and detailed usage.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--tag',
      type=str,
      default='',
      help="""\
      The particular tag to query for. Only used if --inspect is present\
      """
  )
  parser.add_argument(
      '--event_file',
      type=str,
      default='',
      help="""\
      The particular event file to query for. Only used if --inspect is present
      and --logdir is not specified.\
      """
  )
  parser.add_argument(
      '--port',
      type=int,
      default=6006,
      help='What port to serve TensorBoard on.'
  )
  parser.add_argument(
      '--purge_orphaned_data',
      type=bool,
      default=True,
      help="""\
      Whether to purge data that may have been orphaned due to TensorBoard
      restarts. Disabling purge_orphaned_data can be used to debug data
      disappearance.\
      """
  )
  parser.add_argument(
      '--reload_interval',
      type=int,
      default=60,
      help='How often the backend should load more data.'
  )
  FLAGS = parser.parse_args()

  app.run()
