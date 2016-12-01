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
"""Module for building TensorBoard servers.

This is its own module so it can be used in both actual code and test code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import threading
import time
import re

import six
from six.moves import BaseHTTPServer
from six.moves import socketserver

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_accumulator
from tensorflow.python.summary.impl import io_wrapper
from tensorflow.tensorboard.backend import handler

# How many elements to store per tag, by tag type
TENSORBOARD_SIZE_GUIDANCE = {
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 1000,
    event_accumulator.HISTOGRAMS: 50,
}


def ParseEventFilesSpec(logdir):
  """Parses `logdir` into a map from paths to run group names.

  The events files flag format is a comma-separated list of path specifications.
  A path specification either looks like 'group_name:/path/to/directory' or
  '/path/to/directory'; in the latter case, the group is unnamed. Group names
  cannot start with a forward slash: /foo:bar/baz will be interpreted as a
  spec with no name and path '/foo:bar/baz'.

  Globs are not supported.

  Args:
    logdir: A comma-separated list of run specifications.
  Returns:
    A dict mapping directory paths to names like {'/path/to/directory': 'name'}.
    Groups without an explicit name are named after their path. If logdir is
    None, returns an empty dict, which is helpful for testing things that don't
    require any valid runs.
  """
  files = {}
  if logdir is None:
    return files
  # Make sure keeping consistent with ParseURI in core/lib/io/path.cc
  uri_pattern = re.compile("[a-zA-Z][0-9a-zA-Z.]*://.*")
  for specification in logdir.split(','):
    # Check if the spec contains group. A spec start with xyz:// is regarded as
    # URI path spec instead of group spec. If the spec looks like /foo:bar/baz,
    # then we assume it's a path with a colon.
    if uri_pattern.match(specification) is None and \
       ':' in specification and specification[0] != '/':
      # We split at most once so run_name:/path:with/a/colon will work.
      run_name, _, path = specification.partition(':')
    else:
      run_name = None
      path = specification
    if uri_pattern.match(path) is None:
      path = os.path.realpath(path)
    files[path] = run_name
  return files


def ReloadMultiplexer(multiplexer, path_to_run):
  """Loads all runs into the multiplexer.

  Args:
    multiplexer: The `EventMultiplexer` to add runs to and reload.
    path_to_run: A dict mapping from paths to run names, where `None` as the run
      name is interpreted as a run name equal to the path.
  """
  start = time.time()
  logging.info('TensorBoard reload process beginning')
  for (path, name) in six.iteritems(path_to_run):
    multiplexer.AddRunsFromDirectory(path, name)
  logging.info('TensorBoard reload process: Reload the whole Multiplexer')
  multiplexer.Reload()
  duration = time.time() - start
  logging.info('TensorBoard done reloading. Load took %0.3f secs', duration)


def StartMultiplexerReloadingThread(multiplexer, path_to_run, load_interval):
  """Starts a thread to automatically reload the given multiplexer.

  The thread will reload the multiplexer by calling `ReloadMultiplexer` every
  `load_interval` seconds, starting immediately.

  Args:
    multiplexer: The `EventMultiplexer` to add runs to and reload.
    path_to_run: A dict mapping from paths to run names, where `None` as the run
      name is interpreted as a run name equal to the path.
    load_interval: How many seconds to wait after one load before starting the
      next load.

  Returns:
    A started `threading.Thread` that reloads the multiplexer.
  """
  # We don't call multiplexer.Reload() here because that would make
  # AddRunsFromDirectory block until the runs have all loaded.
  def _ReloadForever():
    while True:
      ReloadMultiplexer(multiplexer, path_to_run)
      time.sleep(load_interval)

  thread = threading.Thread(target=_ReloadForever)
  thread.daemon = True
  thread.start()
  return thread


class ThreadedHTTPServer(socketserver.ThreadingMixIn,
                         BaseHTTPServer.HTTPServer):
  """A threaded HTTP server."""
  daemon_threads = True


def BuildServer(multiplexer, host, port, logdir):
  """Sets up an HTTP server for running TensorBoard.

  Args:
    multiplexer: An `EventMultiplexer` that the server will query for
      information about events.
    host: The host name.
    port: The port number to bind to, or 0 to pick one automatically.
    logdir: The logdir argument string that tensorboard started up with.

  Returns:
    A `BaseHTTPServer.HTTPServer`.
  """
  factory = functools.partial(handler.TensorboardHandler, multiplexer, logdir)
  return ThreadedHTTPServer((host, port), factory)
