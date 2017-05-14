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
"""Serve TensorFlow Studio web frontend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import socket

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import logging
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import status_bar
from tensorflow.contrib.studio.core import server

flags.DEFINE_string('working_dir', None, 'Specify working directory.')

flags.DEFINE_boolean('debug', False, 'Whether to run the app in debug mode. '
                     'This increases log verbosity to DEBUG.')

flags.DEFINE_string('host', '0.0.0.0', 'What host to listen to. Defaults to '
                    'serving on 0.0.0.0, set to 127.0.0.1 (localhost) to'
                    'disable remote access (also quiets security warnings).')

flags.DEFINE_integer('port', 6006, 'What port to serve TensorBoard on.')

FLAGS = flags.FLAGS


def main(unused_argv=None):
  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)
    logging.info('Studio is in debug mode.')

  if not FLAGS.working_dir:
    logging.info('A local working dir is used. '
                 'Use --working_dir="path" to specify different one.')
    FLAGS.working_dir = os.getcwd()

  logging.info('Starting Studio with working directory %s', FLAGS.working_dir)

  try:
    studio_server = server.BuildServer(FLAGS.working_dir, FLAGS.host, FLAGS.port)
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

  print('Starting TensorBoard on port %d' % FLAGS.port)
  print('(You can navigate to http://%s:%d)' % (FLAGS.host, FLAGS.port))
  studio_server.serve_forever()


if __name__ == '__main__':
  app.run()
