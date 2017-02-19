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
from werkzeug import serving

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_file_inspector as efi
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import application
from tensorflow.tensorboard.plugins.debugger import plugin as debugger_plugin
from tensorflow.tensorboard.plugins.projector import plugin as projector_plugin

flags.DEFINE_string('logdir', '', """logdir specifies the directory where
TensorBoard will look to find TensorFlow event files that it can display.
TensorBoard will recursively walk the directory structure rooted at logdir,
looking for .*tfevents.* files.

You may also pass a comma separated list of log directories, and TensorBoard
will watch each directory. You can also assign names to individual log
directories by putting a colon between the name and the path, as in

tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
""")

flags.DEFINE_boolean(
    'insecure_debug_mode', False, 'Whether to run the app in debug mode. '
    'This increases log verbosity, and enables debugging on server exceptions.')

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


class Server(object):
  """A simple WSGI-compliant http server that can serve TensorBoard."""

  def get_tag(self):
    """Read the TensorBoard TAG number, and return it or an empty string."""
    try:
      tag = resource_loader.load_resource('tensorboard/TAG').strip()
      logging.info('TensorBoard is tag: %s', tag)
      return tag
    except IOError:
      logging.info('Unable to read TensorBoard tag')
      return ''

  def create_app(self):
    """Creates a WSGI-compliant app than can handle TensorBoard requests.

    Returns:
      (function) A complete WSGI application that handles TensorBoard requests.
    """

    logdir = os.path.expanduser(FLAGS.logdir)
    if not logdir:
      msg = ('A logdir must be specified. Run `tensorboard --help` for '
             'details and examples.')
      logging.error(msg)
      print(msg)
      return -1

    multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=application.DEFAULT_SIZE_GUIDANCE,
        purge_orphaned_data=FLAGS.purge_orphaned_data)
    plugins = {
        debugger_plugin.PLUGIN_PREFIX_ROUTE:
            debugger_plugin.DebuggerPlugin(multiplexer),
        projector_plugin.PLUGIN_PREFIX_ROUTE:
            projector_plugin.ProjectorPlugin(),
    }
    return application.TensorBoardWSGIApp(
        logdir,
        plugins,
        multiplexer,
        reload_interval=FLAGS.reload_interval)

  def serve(self):
    """Starts a WSGI server that serves the TensorBoard app."""

    tb_app = self.create_app()
    logging.info('Starting TensorBoard in directory %s', os.getcwd())
    debug = FLAGS.insecure_debug_mode
    if debug:
      logging.set_verbosity(logging.DEBUG)
      logging.warning('TensorBoard is in debug mode. This is NOT SECURE.')

    print('Starting TensorBoard %s on port %d' % (self.get_tag(), FLAGS.port))
    if FLAGS.host == '0.0.0.0':
      try:
        host = socket.gethostbyname(socket.gethostname())
        print('(You can navigate to http://%s:%d)' % (host, FLAGS.port))
      except socket.gaierror:
        pass
    else:
      print('(You can navigate to http://%s:%d)' % (FLAGS.host, FLAGS.port))

    try:
      serving.run_simple(
          FLAGS.host,
          FLAGS.port,
          tb_app,
          threaded=True,
          use_reloader=debug,
          use_evalex=debug,
          use_debugger=debug)
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


def main(unused_argv=None):
  if FLAGS.inspect:
    logging.info('Not bringing up TensorBoard, but inspecting event files.')
    event_file = os.path.expanduser(FLAGS.event_file)
    efi.inspect(FLAGS.logdir, event_file, FLAGS.tag)
    return 0

  Server().serve()


if __name__ == '__main__':
  app.run()
