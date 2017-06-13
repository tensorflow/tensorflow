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

import tensorflow as tf
from werkzeug import serving


from tensorflow.tensorboard.backend import application
from tensorflow.tensorboard.backend.event_processing import event_file_inspector as efi
from tensorflow.tensorboard.plugins.audio import audio_plugin
from tensorflow.tensorboard.plugins.distributions import distributions_plugin
from tensorflow.tensorboard.plugins.graphs import graphs_plugin
from tensorflow.tensorboard.plugins.histograms import histograms_plugin
from tensorflow.tensorboard.plugins.images import images_plugin
from tensorflow.tensorboard.plugins.projector import projector_plugin
from tensorflow.tensorboard.plugins.scalars import scalars_plugin
from tensorflow.tensorboard.plugins.text import text_plugin

# TensorBoard flags

tf.flags.DEFINE_string('logdir', '', """logdir specifies the directory where
TensorBoard will look to find TensorFlow event files that it can display.
TensorBoard will recursively walk the directory structure rooted at logdir,
looking for .*tfevents.* files.

You may also pass a comma separated list of log directories, and TensorBoard
will watch each directory. You can also assign names to individual log
directories by putting a colon between the name and the path, as in

tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
""")

tf.flags.DEFINE_string(
    'host', '', 'What host to listen to. Defaults to '
    'serving on all interfaces, set to 127.0.0.1 (localhost) to'
    'disable remote access (also quiets security warnings).')

tf.flags.DEFINE_integer('port', 6006, 'What port to serve TensorBoard on.')

tf.flags.DEFINE_boolean(
    'purge_orphaned_data', True, 'Whether to purge data that '
    'may have been orphaned due to TensorBoard restarts. '
    'Disabling purge_orphaned_data can be used to debug data '
    'disappearance.')

tf.flags.DEFINE_integer('reload_interval', 5,
                        'How often the backend should load '
                        'more data.')

# Inspect Mode flags

tf.flags.DEFINE_boolean('inspect', False, """Use this flag to print out a digest
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
tf.flags.DEFINE_string(
    'tag', '',
    'The particular tag to query for. Only used if --inspect is present')
tf.flags.DEFINE_string(
    'event_file', '',
    'The particular event file to query for. Only used if --inspect is present '
    'and --logdir is not specified.')

FLAGS = tf.flags.FLAGS


def create_tb_app(plugins):
  """Read the flags, and create a TensorBoard WSGI application.

  Args:
    plugins: A list of plugins for TensorBoard to initialize.

  Raises:
    ValueError: if a logdir is not specified.

  Returns:
    A new TensorBoard WSGI application.
  """
  if not FLAGS.logdir:
    raise ValueError('A logdir must be specified. Run `tensorboard --help` for '
                     'details and examples.')

  logdir = os.path.expanduser(FLAGS.logdir)
  return application.standard_tensorboard_wsgi(
      logdir=logdir,
      purge_orphaned_data=FLAGS.purge_orphaned_data,
      reload_interval=FLAGS.reload_interval,
      plugins=plugins)


def make_simple_server(tb_app, host, port):
  """Create an HTTP server for TensorBoard.

  Args:
    tb_app: The TensorBoard WSGI application to create a server for.
    host: Indicates the interfaces to bind to ('::' or '0.0.0.0' for all
        interfaces, '::1' or '127.0.0.1' for localhost). A blank value ('')
        indicates protocol-agnostic all interfaces.
    port: The port to bind to (0 indicates an unused port selected by the
        operating system).
  Returns:
    A tuple of (server, url):
      server: An HTTP server object configured to host TensorBoard.
      url: A best guess at a URL where TensorBoard will be accessible once the
        server has been started.
  Raises:
    socket.error: If a server could not be constructed with the host and port
      specified. Also logs an error message.
  """
  # Mute the werkzeug logging.
  base_logging.getLogger('werkzeug').setLevel(base_logging.WARNING)

  try:
    if host:
      # The user gave us an explicit host
      server = serving.make_server(host, port, tb_app, threaded=True)
      if ':' in host and not host.startswith('['):
        # Display IPv6 addresses as [::1]:80 rather than ::1:80
        final_host = '[{}]'.format(host)
      else:
        final_host = host
    else:
      # We've promised to bind to all interfaces on this host. However, we're
      # not sure whether that means IPv4 or IPv6 interfaces.
      try:
        # First try passing in a blank host (meaning all interfaces). This,
        # unfortunately, defaults to IPv4 even if no IPv4 interface is available
        # (yielding a socket.error).
        server = serving.make_server(host, port, tb_app, threaded=True)
      except socket.error:
        # If a blank host didn't work, we explicitly request IPv6 interfaces.
        server = serving.make_server('::', port, tb_app, threaded=True)
      final_host = socket.gethostname()
    server.daemon_threads = True
  except socket.error as socket_error:
    if port == 0:
      msg = 'TensorBoard unable to find any open port'
    else:
      msg = (
          'TensorBoard attempted to bind to port %d, but it was already in use'
          % FLAGS.port)
    tf.logging.error(msg)
    print(msg)
    raise socket_error

  final_port = server.socket.getsockname()[1]
  tensorboard_url = 'http://%s:%d' % (final_host, final_port)
  return server, tensorboard_url


def run_simple_server(tb_app):
  """Run a TensorBoard HTTP server, and print some messages to the console."""
  try:
    server, url = make_simple_server(tb_app, FLAGS.host, FLAGS.port)
  except socket.error:
    # An error message was already logged
    exit(-1)
  msg = 'Starting TensorBoard %s at %s' % (tb_app.tag, url)
  print(msg)
  tf.logging.info(msg)
  print('(Press CTRL+C to quit)')
  sys.stdout.flush()

  server.serve_forever()


def main(unused_argv=None):
  if FLAGS.inspect:
    tf.logging.info('Not bringing up TensorBoard, but inspecting event files.')
    event_file = os.path.expanduser(FLAGS.event_file)
    efi.inspect(FLAGS.logdir, event_file, FLAGS.tag)
    return 0
  else:
    plugins = [
        scalars_plugin.ScalarsPlugin(),
        images_plugin.ImagesPlugin(),
        audio_plugin.AudioPlugin(),
        graphs_plugin.GraphsPlugin(),
        distributions_plugin.DistributionsPlugin(),
        histograms_plugin.HistogramsPlugin(),
        projector_plugin.ProjectorPlugin(),
        text_plugin.TextPlugin(),
    ]
    tb = create_tb_app(plugins)
    run_simple_server(tb)

if __name__ == '__main__':
  tf.app.run()
